from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  vel_norm_sqrt = torch.sqrt(vel_norm + 1e-6)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm_sqrt, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.peak_heights[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


class swing_quality_reward:
  """Weak positive [0,1] reward for the quality of completed swings.

  The reward is evaluated only on landing events, so it shapes how feet swing
  without strongly incentivizing the policy to increase gait frequency.

  Clearance is scored against a target peak height, optionally with asymmetric
  penalties so under-clearance can be punished more strongly than over-clearance.
  Air time is scored against a target swing duration. The per-step reward is the
  mean landing quality across feet that landed on that step, and zero otherwise.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor_name: str | None = cfg.params.get("height_sensor_name")
    if height_sensor_name is not None:
      height_sensor = env.scene[height_sensor_name]
      assert isinstance(height_sensor, TerrainHeightSensor), (
        f"swing_quality_reward requires a TerrainHeightSensor, "
        f"got {type(height_sensor).__name__}"
      )
      n_feet = height_sensor.num_frames
    else:
      n_feet = len(cfg.params["asset_cfg"].site_names)

    self.n_feet = n_feet
    self.step_dt = env.step_dt

    batch = env.num_envs
    device = env.device
    self.peak_heights = torch.zeros(batch, n_feet, device=device)
    self.swing_air_time = torch.zeros(batch, n_feet, device=device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.peak_heights[env_ids] = 0.0
    self.swing_air_time[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    target_air_time: float,
    clearance_std: float,
    air_time_std: float,
    command_name: str,
    command_threshold: float,
    height_sensor_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    over_clearance_scale: float = 1.0,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]

    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = (linear_norm + angular_norm) > command_threshold  # [B]
    active_feet = active.unsqueeze(1)

    inactive_feet = ~active_feet
    self.peak_heights = torch.where(
      inactive_feet,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    self.swing_air_time = torch.where(
      inactive_feet,
      torch.zeros_like(self.swing_air_time),
      self.swing_air_time,
    )

    if height_sensor_name is not None:
      height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
      foot_z = height_sensor.data.heights  # [B, F]
    else:
      asset: Entity = env.scene[asset_cfg.name]
      site_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, F]
      foot_z = site_z - site_z.min(dim=1, keepdim=True).values

    assert contact_sensor.data.found is not None
    in_air = contact_sensor.data.found == 0  # [B, N]
    if in_air.shape[1] != self.n_feet or foot_z.shape[1] != self.n_feet:
      raise ValueError(
        "swing_quality_reward requires matching foot counts across contact and height inputs."
      )

    tracked_in_air = in_air & active_feet
    self.swing_air_time = torch.where(
      tracked_in_air,
      self.swing_air_time + self.step_dt,
      self.swing_air_time,
    )
    self.peak_heights = torch.where(
      tracked_in_air,
      torch.maximum(self.peak_heights, foot_z),
      self.peak_heights,
    )

    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    tracked_landing = first_contact & active_feet
    landing = tracked_landing.float()

    clearance_error = self.peak_heights - target_height
    if over_clearance_scale <= 0.0:
      raise ValueError("swing_quality_reward over_clearance_scale must be positive.")
    scaled_clearance_error = torch.where(
      clearance_error >= 0.0,
      clearance_error / over_clearance_scale,
      clearance_error,
    )
    clearance_quality = torch.exp(
      -torch.square(scaled_clearance_error) / (clearance_std**2)
    )
    air_time_quality = torch.exp(
      -torch.square(self.swing_air_time - target_air_time) / (air_time_std**2)
    )
    swing_quality = clearance_quality * air_time_quality

    landing_count = landing.sum(dim=1)
    landing_quality = (swing_quality * landing).sum(dim=1) / torch.clamp(
      landing_count, min=1.0
    )
    reward = torch.where(
      landing_count > 0, landing_quality, torch.zeros_like(landing_count)
    )

    log = env.extras.setdefault("log", {})
    total_landings = landing.sum().clamp(min=1.0)
    log["Metrics/swing_quality_peak_height"] = (
      self.peak_heights * landing
    ).sum() / total_landings
    log["Metrics/swing_quality_air_time"] = (
      self.swing_air_time * landing
    ).sum() / total_landings
    log["Metrics/swing_quality_mean"] = (swing_quality * landing).sum() / total_landings

    reset_mask = tracked_landing | inactive_feet
    self.peak_heights = torch.where(
      reset_mask,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    self.swing_air_time = torch.where(
      reset_mask,
      torch.zeros_like(self.swing_air_time),
      self.swing_air_time,
    )

    return reward * active.float()


class gait_reward:
  """Positive [0,1] reward for gait quality (clearance + air time).

  Tracks peak clearance and air time during each foot's swing phase. At
  landing, computes a quality score for the completed swing.

  Scores persist through stance and only decay for feet that stay in swing
  without landing. This keeps the reward dense while avoiding a strong bias
  toward rapid footfall cadence.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor_name: str | None = cfg.params.get("height_sensor_name")
    if height_sensor_name is not None:
      height_sensor = env.scene[height_sensor_name]
      assert isinstance(height_sensor, TerrainHeightSensor), (
        f"gait_reward requires a TerrainHeightSensor, "
        f"got {type(height_sensor).__name__}"
      )
      n_feet = height_sensor.num_frames
    else:
      n_feet = len(cfg.params["asset_cfg"].site_names)

    self.n_feet = n_feet
    self.step_dt = env.step_dt

    B = env.num_envs
    device = env.device
    self.peak_heights = torch.zeros(B, n_feet, device=device)
    self.swing_air_time = torch.zeros(B, n_feet, device=device)
    self.foot_scores = torch.zeros(B, n_feet, device=device)

    halflife = cfg.params.get("score_halflife", 0.5)
    if halflife <= 0.0:
      raise ValueError("gait_reward score_halflife must be positive.")
    self.decay = 0.5 ** (self.step_dt / halflife)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.peak_heights[env_ids] = 0.0
    self.swing_air_time[env_ids] = 0.0
    self.foot_scores[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    target_air_time: float,
    clearance_std: float,
    air_time_std: float,
    command_name: str,
    command_threshold: float,
    height_sensor_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    score_halflife: float = 0.5,
  ) -> torch.Tensor:
    del score_halflife  # Consumed in __init__.
    contact_sensor: ContactSensor = env.scene[sensor_name]

    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = (linear_norm + angular_norm) > command_threshold  # [B]
    active_feet = active.unsqueeze(1)

    # Reset inactive envs so reward state does not carry stale gait history.
    inactive_feet = ~active_feet
    self.peak_heights = torch.where(
      inactive_feet,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    self.swing_air_time = torch.where(
      inactive_feet,
      torch.zeros_like(self.swing_air_time),
      self.swing_air_time,
    )
    self.foot_scores = torch.where(
      inactive_feet,
      torch.zeros_like(self.foot_scores),
      self.foot_scores,
    )

    # Get foot heights: terrain-relative if sensor available, else relative to the
    # lowest foot in the batch as a flat-ground fallback.
    if height_sensor_name is not None:
      height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
      foot_z = height_sensor.data.heights  # [B, F]
    else:
      asset: Entity = env.scene[asset_cfg.name]
      site_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, F]
      foot_z = site_z - site_z.min(dim=1, keepdim=True).values

    # Swing / stance.
    assert contact_sensor.data.found is not None
    in_air = contact_sensor.data.found == 0  # [B, N]
    if in_air.shape[1] != self.n_feet or foot_z.shape[1] != self.n_feet:
      raise ValueError(
        "gait_reward requires matching foot counts across contact and height inputs."
      )

    tracked_in_air = in_air & active_feet

    # Accumulate swing metrics while in air.
    self.swing_air_time = torch.where(
      tracked_in_air,
      self.swing_air_time + self.step_dt,
      self.swing_air_time,
    )
    self.peak_heights = torch.where(
      tracked_in_air,
      torch.maximum(self.peak_heights, foot_z),
      self.peak_heights,
    )

    # Detect landing events.
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    tracked_landing = first_contact & active_feet
    landing = tracked_landing.float()  # [B, N]

    # Quality of the completed swing.
    clearance_error = self.peak_heights - target_height
    clearance_quality = torch.exp(-torch.square(clearance_error) / (clearance_std**2))
    air_time_quality = torch.exp(
      -torch.square(self.swing_air_time - target_air_time) / (air_time_std**2)
    )
    swing_quality = clearance_quality * air_time_quality  # [B, N]

    # Log metrics.
    n_landings = landing.sum().clamp(min=1.0)
    log = env.extras.setdefault("log", {})
    log["Metrics/gait_peak_height"] = (self.peak_heights * landing).sum() / n_landings
    log["Metrics/gait_air_time"] = (self.swing_air_time * landing).sum() / n_landings
    log["Metrics/gait_swing_quality"] = (swing_quality * landing).sum() / n_landings

    # Refresh at landing, decay only for feet that remain in swing without landing,
    # and preserve scores during stance.
    decaying = tracked_in_air & ~tracked_landing
    decayed_scores = torch.where(
      decaying, self.foot_scores * self.decay, self.foot_scores
    )
    self.foot_scores = torch.where(tracked_landing, swing_quality, decayed_scores)

    # Reset swing tracking for landed or inactive feet.
    reset_mask = tracked_landing | inactive_feet
    self.peak_heights = torch.where(
      reset_mask,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    self.swing_air_time = torch.where(
      reset_mask,
      torch.zeros_like(self.swing_air_time),
      self.swing_air_time,
    )

    # Reward = mean foot score, bounded [0, 1].
    reward = self.foot_scores.mean(dim=1)
    return reward * active.float()
