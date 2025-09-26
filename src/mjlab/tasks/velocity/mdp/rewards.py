from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def subtree_angmom_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_name: str = "robot",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_name]
  if sensor_name not in asset.sensor_names:
    raise ValueError(
      f"Sensor '{sensor_name}' not found in asset '{asset_name}'. "
      f"Available sensors: {asset.sensor_names}"
    )
  angmom_w = asset.data.sensor_data[sensor_name]
  angmom_xy_w = angmom_w[:, :2]
  return torch.sum(torch.square(angmom_xy_w), dim=1)


class feet_air_time:
  """Reward long steps taken by the feet.

  This rewards the agent for lifting feet off the ground for longer than a threshold.
  Provides continuous reward signal during flight phase and smooth command scaling.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.threshold_min = cfg.params["threshold_min"]
    self.threshold_max = cfg.params.get("threshold_max", self.threshold_min + 0.3)
    self.asset_name = cfg.params["asset_name"]
    self.sensor_names = cfg.params["sensor_names"]
    self.num_feet = len(self.sensor_names)
    self.command_name = cfg.params["command_name"]
    self.command_threshold = cfg.params["command_threshold"]
    self.reward_mode = cfg.params.get("reward_mode", "continuous")
    self.command_scale_type = cfg.params.get("command_scale_type", "smooth")
    self.command_scale_width = cfg.params.get("command_scale_width", 0.2)

    asset: Entity = env.scene[self.asset_name]
    for sensor_name in self.sensor_names:
      if sensor_name not in asset.sensor_names:
        raise ValueError(
          f"Sensor '{sensor_name}' not found in asset '{self.asset_name}'"
        )

    self.current_air_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)
    self.current_contact_time = torch.zeros(
      env.num_envs, self.num_feet, device=env.device
    )
    self.last_air_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    asset: Entity = env.scene[self.asset_name]

    contact_list = []
    for sensor_name in self.sensor_names:
      sensor_data = asset.data.sensor_data[sensor_name]
      foot_contact = sensor_data[:, 0] > 0
      contact_list.append(foot_contact)

    in_contact = torch.stack(contact_list, dim=1)
    in_air = ~in_contact

    # Detect first contact (landing).
    first_contact = (self.current_air_time > 0) & in_contact

    # Save air time when landing.
    self.last_air_time = torch.where(
      first_contact, self.current_air_time, self.last_air_time
    )

    # Update air time and contact time.
    self.current_air_time = torch.where(
      in_contact,
      torch.zeros_like(self.current_air_time),  # Reset when in contact.
      self.current_air_time + env.step_dt,  # Increment when in air.
    )

    self.current_contact_time = torch.where(
      in_contact,
      self.current_contact_time + env.step_dt,  # Increment when in contact.
      torch.zeros_like(self.current_contact_time),  # Reset when in air.
    )

    if self.reward_mode == "continuous":
      # Give constant reward of 1.0 for each foot that's in air and above threshold.
      exceeds_min = self.current_air_time > self.threshold_min
      below_max = self.current_air_time <= self.threshold_max
      reward_per_foot = torch.where(
        in_air & exceeds_min & below_max,
        torch.ones_like(self.current_air_time),
        torch.zeros_like(self.current_air_time),
      )
      reward = torch.sum(reward_per_foot, dim=1)
    else:
      # This mode gives (air_time - threshold) as reward on landing.
      air_time_over_min = (self.last_air_time - self.threshold_min).clamp(min=0.0)
      air_time_clamped = air_time_over_min.clamp(
        max=self.threshold_max - self.threshold_min
      )
      reward = torch.sum(air_time_clamped * first_contact, dim=1) / env.step_dt

    command = env.command_manager.get_command(self.command_name)
    assert command is not None
    command_norm = torch.norm(command[:, :2], dim=1)
    if self.command_scale_type == "smooth":
      scale = 0.5 * (
        1.0
        + torch.tanh((command_norm - self.command_threshold) / self.command_scale_width)
      )
      reward *= scale
    else:
      reward *= command_norm > self.command_threshold
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.current_air_time[env_ids] = 0.0
    self.current_contact_time[env_ids] = 0.0
    self.last_air_time[env_ids] = 0.0


class gait_smoothness:
  """Penalize jerky, non-smooth gait patterns via joint jerk."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.asset_name = cfg.params.get("asset_name", "robot")
    self.smoothness_weight = cfg.params.get("smoothness_weight", 1.0)

    asset = env.scene[self.asset_name]
    num_joints = asset.data.joint_pos.shape[1]
    self.last_joint_acc = torch.zeros(env.num_envs, num_joints, device=env.device)

    self.dt = env.step_dt

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    asset = env.scene[self.asset_name]

    current_joint_acc = asset.data.joint_acc
    joint_jerk = torch.abs(current_joint_acc - self.last_joint_acc) / self.dt
    total_jerk = torch.sum(joint_jerk, dim=1) * self.smoothness_weight
    self.last_joint_acc = current_joint_acc.clone()
    return total_jerk

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.last_joint_acc[env_ids] = 0.0


class cost_of_transport:
  """Penalize energy expenditure per unit distance traveled."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.command_name = cfg.params.get("command_name", "twist")
    self.asset_name = cfg.params.get("asset_name", "robot")
    self.min_velocity = cfg.params.get("min_velocity", 0.1)  # Avoid division by zero.
    self.normalize_by_mass = cfg.params.get("normalize_by_mass", True)
    self.power_scale = cfg.params.get("power_scale", 0.001)

    if self.normalize_by_mass:
      asset: Entity = env.scene[self.asset_name]
      self.robot_mass = asset.data.model.body_subtreemass[asset.indexing.root_body_id]
    else:
      self.robot_mass = 1.0

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    asset: Entity = env.scene[self.asset_name]

    # Compute electrical power (W).
    tau = asset.data.actuator_force
    qd = asset.data.joint_vel
    mech = tau * qd
    mech_pos = torch.clamp(mech, min=0.0)  # Don't penalize regen.
    total_power = torch.sum(mech_pos, dim=1)  # Watts.

    # Get forward velocity (m/s).
    forward_vel = asset.data.root_link_lin_vel_b[:, 0]
    vel_magnitude = torch.abs(forward_vel)
    vel_clamped = torch.clamp(vel_magnitude, min=self.min_velocity)

    if self.normalize_by_mass:
      # Dimensionless CoT: power / (mass * g * velocity).
      cost = total_power / (self.robot_mass * 9.81 * vel_clamped)
    else:
      # Energy per meter (J/m).
      cost = total_power / vel_clamped

    cost_scaled = cost * self.power_scale

    command = env.command_manager.get_command(self.command_name)
    if command is not None:
      is_moving_command = torch.norm(command[:, :2], dim=1) > 0.1
      cost_scaled = torch.where(
        is_moving_command, cost_scaled, torch.zeros_like(cost_scaled)
      )

    return cost_scaled

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    del env_ids  # Unused.
