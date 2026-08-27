from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import ball_pos_b

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def user_command_linear_velocity_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Planar base-velocity error relative to the unmodified user command."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  robot: Entity = env.scene[asset_cfg.name]
  return torch.linalg.vector_norm(
    robot.data.root_link_lin_vel_b[:, :2] - command[:, :2], dim=1
  )


def user_command_yaw_velocity_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute yaw-rate error relative to the unmodified user command."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  robot: Entity = env.scene[asset_cfg.name]
  return torch.abs(robot.data.root_link_ang_vel_b[:, 2] - command[:, 2])


def command_velocity_envelope_violation(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_tolerance_x: float = 0.10,
  min_tolerance_y: float = 0.08,
  min_tolerance_yaw: float = 0.15,
  relative_tolerance: float = 0.20,
  smoothing_alpha: float = 0.10,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Fraction of smoothed velocity components outside the recovery envelope.

  The short EMA removes gait-cycle oscillations from the curriculum gate.  The
  reward itself remains instantaneous, so this relaxation affects progression
  measurement only and does not weaken the learned speed constraint.
  """
  if not 0.0 < smoothing_alpha <= 1.0:
    raise ValueError("smoothing_alpha must be in (0, 1]")
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
  cache_key = "_football_command_velocity_metric_ema"
  smoothed = vars(env).get(cache_key)
  if not isinstance(smoothed, torch.Tensor) or smoothed.shape != actual.shape:
    smoothed = actual.clone()
    vars(env)[cache_key] = smoothed
  reset = env.episode_length_buf == 0
  smoothed[reset] = actual[reset]
  smoothed[~reset].lerp_(actual[~reset], smoothing_alpha)
  minimum = torch.tensor(
    (min_tolerance_x, min_tolerance_y, min_tolerance_yaw),
    device=command.device,
    dtype=command.dtype,
  )
  tolerance = torch.maximum(minimum, relative_tolerance * torch.abs(command))
  component_violation = torch.abs(smoothed - command) > tolerance
  return component_violation.float().mean(dim=1)


def ball_control_zone_success(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float] = (0.05, 0.45),
  y_abs: float = 0.15,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """One while the physical football is inside the robot control zone."""
  relative = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
  in_x = (relative[:, 0] >= x_range[0]) & (relative[:, 0] <= x_range[1])
  in_y = torch.abs(relative[:, 1]) <= y_abs
  return (in_x & in_y).float()
