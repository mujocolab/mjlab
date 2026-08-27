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
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

from .observations import ball_pos_b

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def _get_velocity_command(
  env: ManagerBasedRlEnv,
  command_name: str,
  use_user_command: bool,
  use_ball_command: bool,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  if use_user_command and use_ball_command:
    raise ValueError("Cannot use both user and football-specific commands.")
  if not use_user_command and not use_ball_command:
    return command
  command_term = env.command_manager.get_term(command_name)
  attribute = "ball_command" if use_ball_command else "user_command"
  selected_command = getattr(command_term, attribute, None)
  if selected_command is None:
    raise ValueError(f"Command '{command_name}' does not expose {attribute!r}.")
  return selected_command


def _football_visibility_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return the shared, smoothed Actor football-visibility gate."""
  cache = vars(env).get("_football_masked_ball_visual")
  if isinstance(cache, dict):
    gate = cache.get("visibility_gate")
    if isinstance(gate, torch.Tensor) and gate.shape == (env.num_envs,):
      return gate
    visible = cache.get("visible")
    if isinstance(visible, torch.Tensor) and visible.shape == (env.num_envs, 1):
      return visible[:, 0]
  return torch.zeros(env.num_envs, device=env.device)


def _football_sensor_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return the smoothed health gate for exogenous ball-sensor dropout."""
  cache = vars(env).get("_football_masked_ball_visual")
  if isinstance(cache, dict):
    gate = cache.get("sensor_gate")
    if isinstance(gate, torch.Tensor) and gate.shape == (env.num_envs,):
      return gate
  return torch.ones(env.num_envs, device=env.device)


class track_ball_lin_vel_xy_exp:
  """Reward instantaneous football planar velocity tracking."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    std = cfg.params["std"]
    if std <= 0.0:
      raise ValueError(f"std must be positive, got {std}")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    control_x_range: tuple[float, float] = (0.05, 0.45),
    control_y_abs: float = 0.15,
    gate_std_x: float = 0.10,
    gate_std_y: float = 0.05,
    gate_by_position: bool = True,
    gate_by_visibility: bool = False,
    gate_by_sensor_health: bool = False,
    use_user_command: bool = False,
    use_ball_command: bool = False,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    ball: Entity = env.scene[ball_cfg.name]
    robot: Entity = env.scene[asset_cfg.name]
    command = _get_velocity_command(
      env,
      command_name,
      use_user_command,
      use_ball_command,
    )

    ball_velocity_b = quat_apply_inverse(
      robot.data.root_link_quat_w, ball.data.root_link_lin_vel_w
    )

    error = torch.sum(torch.square(command[:, :2] - ball_velocity_b[:, :2]), dim=1)
    velocity_reward = torch.exp(-error / std**2)
    if gate_by_position:
      x_min, x_max = control_x_range
      if x_min > x_max or control_y_abs <= 0.0:
        raise ValueError("Invalid football control-zone bounds")
      if gate_std_x <= 0.0 or gate_std_y <= 0.0:
        raise ValueError("gate_std_x and gate_std_y must be positive")
      ball_relative_b = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
      x_out = torch.relu(
        torch.maximum(x_min - ball_relative_b[:, 0], ball_relative_b[:, 0] - x_max)
        / gate_std_x
      )
      y_out = torch.relu(
        (torch.abs(ball_relative_b[:, 1]) - control_y_abs) / gate_std_y
      )
      velocity_reward *= torch.exp(-(x_out.square() + y_out.square()))
    if gate_by_visibility:
      velocity_reward *= _football_visibility_gate(env)
    if gate_by_sensor_health:
      velocity_reward *= _football_sensor_gate(env)
    return velocity_reward


def stop_ball_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  command_threshold: float = 0.1,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
  """Reward low football planar speed during low-speed commands only."""
  if std <= 0.0:
    raise ValueError(f"std must be positive, got {std}")
  if command_threshold < 0.0:
    raise ValueError(f"command_threshold must be non-negative, got {command_threshold}")

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  ball: Entity = env.scene[ball_cfg.name]
  command_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
  ball_speed_error = torch.sum(
    torch.square(ball.data.root_link_lin_vel_w[:, :2]), dim=1
  )
  stop_reward = torch.exp(-ball_speed_error / std**2)
  return stop_reward * (command_speed < command_threshold).float()


class track_ball_relative_vel_xy_exp:
  """Reward low instantaneous ball-to-pelvis planar velocity."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    std = cfg.params["std"]
    if std <= 0.0:
      raise ValueError(f"std must be positive, got {std}")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    period: float | None = None,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del period
    ball: Entity = env.scene[ball_cfg.name]
    robot: Entity = env.scene[asset_cfg.name]
    relative_velocity_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    relative_velocity_yaw = quat_apply_inverse(
      yaw_quat(robot.data.root_link_quat_w), relative_velocity_w
    )

    error = torch.sum(torch.square(relative_velocity_yaw[:, :2]), dim=1)
    return torch.exp(-error / std**2)

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    del env_ids


def track_ball_relative_pos_xy_exp(
  env: ManagerBasedRlEnv,
  std_x: float,
  std_y: float,
  command_name: str,
  anchor_x: float,
  anchor_x_speed_gain: float,
  anchor_x_range: tuple[float, float],
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the ball for staying near a speed-conditioned pelvis-frame anchor."""
  if std_x <= 0.0 or std_y <= 0.0:
    raise ValueError(f"std_x and std_y must be positive, got {std_x}, {std_y}")
  anchor_x_min, anchor_x_max = anchor_x_range
  if anchor_x_min > anchor_x_max:
    raise ValueError(f"anchor_x_range must be ordered, got {anchor_x_range}")

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  ball_relative_yaw = ball_pos_b(
    env,
    ball_cfg=ball_cfg,
    asset_cfg=asset_cfg,
  )
  command_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
  target_x = torch.clamp(
    anchor_x + anchor_x_speed_gain * command_speed,
    min=anchor_x_min,
    max=anchor_x_max,
  )
  x_error = torch.square((ball_relative_yaw[:, 0] - target_x) / std_x)
  y_error = torch.square(ball_relative_yaw[:, 1] / std_y)
  return torch.exp(-(x_error + y_error))


def ball_front_control(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float],
  y_abs: float,
  gate_by_visibility: bool = False,
  gate_by_sensor_health: bool = False,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping the football center inside a hard robot-frame control zone."""
  x_min, x_max = x_range
  if x_min > x_max:
    raise ValueError(f"x_range must be ordered, got {x_range}")
  if y_abs <= 0.0:
    raise ValueError(f"y_abs must be positive, got {y_abs}")

  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[asset_cfg.name]
  ball_relative_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  ball_pos_b = quat_apply_inverse(robot.data.root_link_quat_w, ball_relative_w)
  in_x_range = (ball_pos_b[:, 0] >= x_min) & (ball_pos_b[:, 0] <= x_max)
  in_y_range = torch.abs(ball_pos_b[:, 1]) <= y_abs
  reward = (in_x_range & in_y_range).float()
  if gate_by_visibility:
    reward *= _football_visibility_gate(env)
  if gate_by_sensor_health:
    reward *= _football_sensor_gate(env)
  return reward


def track_visibility_blended_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  target_ball_x: float = 0.25,
  recovery_gain_x: float = 1.0,
  recovery_gain_y: float = 1.5,
  min_tolerance_x: float = 0.10,
  min_tolerance_y: float = 0.08,
  relative_tolerance: float = 0.20,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track command velocity plus a visibility-gated, bounded ball correction."""
  if std <= 0.0:
    raise ValueError("std must be positive")
  if min_tolerance_x <= 0.0 or min_tolerance_y <= 0.0:
    raise ValueError("minimum velocity tolerances must be positive")
  if relative_tolerance < 0.0:
    raise ValueError("relative_tolerance must be non-negative")

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  robot: Entity = env.scene[asset_cfg.name]
  ball_relative = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
  gate = _football_visibility_gate(env)
  tolerance_x = torch.maximum(
    torch.full_like(command[:, 0], min_tolerance_x),
    relative_tolerance * torch.abs(command[:, 0]),
  )
  tolerance_y = torch.maximum(
    torch.full_like(command[:, 1], min_tolerance_y),
    relative_tolerance * torch.abs(command[:, 1]),
  )
  correction_x = torch.clamp(
    recovery_gain_x * (ball_relative[:, 0] - target_ball_x),
    min=-tolerance_x,
    max=tolerance_x,
  )
  correction_y = torch.clamp(
    recovery_gain_y * ball_relative[:, 1],
    min=-tolerance_y,
    max=tolerance_y,
  )
  target_xy = command[:, :2] + gate[:, None] * torch.stack(
    (correction_x, correction_y), dim=1
  )
  actual = robot.data.root_link_lin_vel_b
  error = torch.sum(torch.square(target_xy - actual[:, :2]), dim=1)
  error += torch.square(actual[:, 2])
  return torch.exp(-error / std**2)


def track_visibility_blended_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  recovery_gain_yaw: float = 1.5,
  min_tolerance_yaw: float = 0.15,
  relative_tolerance: float = 0.20,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track yaw command plus a bounded correction toward a visible football."""
  if std <= 0.0 or min_tolerance_yaw <= 0.0:
    raise ValueError("std and min_tolerance_yaw must be positive")
  if relative_tolerance < 0.0:
    raise ValueError("relative_tolerance must be non-negative")

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  robot: Entity = env.scene[asset_cfg.name]
  ball_relative = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
  gate = _football_visibility_gate(env)
  tolerance = torch.maximum(
    torch.full_like(command[:, 2], min_tolerance_yaw),
    relative_tolerance * torch.abs(command[:, 2]),
  )
  bearing = torch.atan2(ball_relative[:, 1], ball_relative[:, 0])
  correction = torch.clamp(
    recovery_gain_yaw * bearing,
    min=-tolerance,
    max=tolerance,
  )
  target_yaw = command[:, 2] + gate * correction
  actual = robot.data.root_link_ang_vel_b
  error = torch.square(target_yaw - actual[:, 2])
  error += torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-error / std**2)


def track_visible_recovery_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  target_ball_x: float = 0.25,
  recovery_gain_x: float = 1.0,
  recovery_gain_y: float = 1.5,
  min_tolerance_x: float = 0.10,
  min_tolerance_y: float = 0.08,
  relative_tolerance: float = 0.20,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Visible-branch reward for bounded recovery toward the football."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  robot: Entity = env.scene[asset_cfg.name]
  ball_relative = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
  gate = _football_visibility_gate(env)
  tolerance_x = torch.maximum(
    torch.full_like(command[:, 0], min_tolerance_x),
    relative_tolerance * torch.abs(command[:, 0]),
  )
  tolerance_y = torch.maximum(
    torch.full_like(command[:, 1], min_tolerance_y),
    relative_tolerance * torch.abs(command[:, 1]),
  )
  correction = torch.stack(
    (
      torch.clamp(
        recovery_gain_x * (ball_relative[:, 0] - target_ball_x),
        min=-tolerance_x,
        max=tolerance_x,
      ),
      torch.clamp(
        recovery_gain_y * ball_relative[:, 1],
        min=-tolerance_y,
        max=tolerance_y,
      ),
    ),
    dim=1,
  )
  target = command[:, :2] + correction
  actual = robot.data.root_link_lin_vel_b
  error = torch.sum(torch.square(target - actual[:, :2]), dim=1)
  error += torch.square(actual[:, 2])
  return gate * torch.exp(-error / std**2)


def track_visible_recovery_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  recovery_gain_yaw: float = 1.5,
  min_tolerance_yaw: float = 0.15,
  relative_tolerance: float = 0.20,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Visible-branch yaw reward for a bounded turn toward the football."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  robot: Entity = env.scene[asset_cfg.name]
  ball_relative = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
  gate = _football_visibility_gate(env)
  tolerance = torch.maximum(
    torch.full_like(command[:, 2], min_tolerance_yaw),
    relative_tolerance * torch.abs(command[:, 2]),
  )
  correction = torch.clamp(
    recovery_gain_yaw * torch.atan2(ball_relative[:, 1], ball_relative[:, 0]),
    min=-tolerance,
    max=tolerance,
  )
  actual = robot.data.root_link_ang_vel_b
  error = torch.square(command[:, 2] + correction - actual[:, 2])
  error += torch.sum(torch.square(actual[:, :2]), dim=1)
  return gate * torch.exp(-error / std**2)


def track_hidden_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Hidden-branch reward for following the unmodified user command."""
  return (1.0 - _football_visibility_gate(env)) * track_linear_velocity(
    env, std=std, command_name=command_name, asset_cfg=asset_cfg
  )


def track_hidden_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Hidden-branch yaw reward for following the unmodified user command."""
  return (1.0 - _football_visibility_gate(env)) * track_angular_velocity(
    env, std=std, command_name=command_name, asset_cfg=asset_cfg
  )


def command_velocity_envelope_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_tolerance_x: float = 0.10,
  min_tolerance_y: float = 0.08,
  min_tolerance_yaw: float = 0.15,
  relative_tolerance: float = 0.20,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic cost only outside the command-relative recovery envelope."""
  if min(min_tolerance_x, min_tolerance_y, min_tolerance_yaw) <= 0.0:
    raise ValueError("minimum velocity tolerances must be positive")
  if relative_tolerance < 0.0:
    raise ValueError("relative_tolerance must be non-negative")
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  robot: Entity = env.scene[asset_cfg.name]
  actual = torch.stack(
    (
      robot.data.root_link_lin_vel_b[:, 0],
      robot.data.root_link_lin_vel_b[:, 1],
      robot.data.root_link_ang_vel_b[:, 2],
    ),
    dim=1,
  )
  minimum = torch.tensor(
    (min_tolerance_x, min_tolerance_y, min_tolerance_yaw),
    device=command.device,
    dtype=command.dtype,
  )
  tolerance = torch.maximum(minimum, relative_tolerance * torch.abs(command))
  excess = torch.relu(torch.abs(actual - command) - tolerance)
  return torch.sum(torch.square(excess), dim=1)


def ball_outside_control_zone_l2(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float],
  y_abs: float,
  std_x: float,
  std_y: float,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return a soft hinge cost when the ball leaves the pelvis control zone.

  The cost is zero inside ``x_range`` and ``|y| <= y_abs`` and grows
  quadratically outside the corresponding boundary.  The reward term should
  therefore use a negative weight.
  """
  x_min, x_max = x_range
  if x_min > x_max:
    raise ValueError(f"x_range must be ordered, got {x_range}")
  if y_abs <= 0.0 or std_x <= 0.0 or std_y <= 0.0:
    raise ValueError("y_abs, std_x, and std_y must be positive")

  ball_relative_b = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
  x_low = torch.relu((x_min - ball_relative_b[:, 0]) / std_x)
  x_high = torch.relu((ball_relative_b[:, 0] - x_max) / std_x)
  y_out = torch.relu((torch.abs(ball_relative_b[:, 1]) - y_abs) / std_y)
  return x_low.square() + x_high.square() + y_out.square()


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


def klavier_track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """IsaacLab/Klavier planar velocity tracking reward."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-error / std**2)


def klavier_track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """IsaacLab/Klavier yaw-rate tracking reward."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / std**2)


def klavier_lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def klavier_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def klavier_body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  gravity_b = quat_apply_inverse(quat_w, asset.data.gravity_vec_w)
  return torch.sum(torch.square(gravity_b[:, :2]), dim=1)


def klavier_joint_mirror(
  env: ManagerBasedRlEnv,
  mirror_joints: tuple[tuple[str, str], ...],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reproduce the two cross-body posture pairs used by the source task."""
  asset: Entity = env.scene[asset_cfg.name]
  cache_name = "_klavier_joint_mirror_ids"
  pairs = getattr(env, cache_name, None)
  if pairs is None:
    pairs = []
    for left, right in mirror_joints:
      left_ids, _ = asset.find_joints(left)
      right_ids, _ = asset.find_joints(right)
      pairs.append((left_ids, right_ids))
    setattr(env, cache_name, pairs)
  result = torch.zeros(env.num_envs, device=env.device)
  default = asset.data.default_joint_pos
  assert default is not None
  for left_ids, right_ids in pairs:
    left = asset.data.joint_pos[:, left_ids] - default[:, left_ids]
    right = asset.data.joint_pos[:, right_ids] - default[:, right_ids]
    result += torch.sum(torch.square(left - right), dim=1)
  return result / max(len(pairs), 1)


def klavier_joint_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(error), dim=1)


def klavier_joint_deviation_exp(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  return torch.exp(-torch.sum(torch.square(error), dim=1) / std**2)


def klavier_feet_gait(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  period: float,
  offset: tuple[float, float],
  threshold: float,
  command_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  contact_time = sensor.data.current_contact_time
  assert contact_time is not None
  is_contact = contact_time > 0
  phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.tensor(offset, device=env.device).view(1, -1)
  is_stance = ((phase + offsets) % 1.0) < threshold
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (is_stance == is_contact).float().mean(dim=1) * (
    torch.linalg.norm(command, dim=1) > 0.1
  )


def klavier_feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.current_air_time
  contact = sensor.data.current_contact_time
  assert air is not None and contact is not None
  in_contact = contact > 0
  mode_time = torch.where(in_contact, contact, air)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(
    torch.where(single_stance.unsqueeze(-1), mode_time, 0.0), dim=1
  ).values
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.clamp(threshold - torch.abs(mode_time - threshold), min=0.0) * (
    torch.linalg.norm(command, dim=1) > 0.1
  )


def klavier_feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.linalg.norm(
    asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
  )
  return torch.sum(speed * (found > 0), dim=1)


def klavier_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  return torch.sum(
    torch.clamp(torch.linalg.norm(force, dim=-1) - threshold, min=0.0), dim=1
  )


def klavier_undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  return (torch.linalg.norm(force, dim=-1) > threshold).any(dim=1).float()


def klavier_stand_still_without_cmd(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.sum(torch.abs(diff), dim=1) * (
    torch.linalg.norm(command, dim=1) < 0.1
  )


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
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
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
