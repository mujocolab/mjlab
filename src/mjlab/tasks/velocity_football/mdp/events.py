"""Reset and randomization events for the velocity-football task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def reset_football(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  ball_radius: float = 0.1098,
  robot_xy_noise_range: tuple[float, float] = (-0.05, 0.05),
  robot_yaw_range: tuple[float, float] = (-3.14, 3.14),
  ball_forward_range: tuple[float, float] = (0.1, 0.5),
  ball_lateral_range: tuple[float, float] = (-0.15, 0.15),
  ball_velocity_range: tuple[float, float] = (-1.5, 1.5),
) -> None:
  """Reset the robot and football in a consistent robot-relative arrangement."""
  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  num_envs = len(env_ids)
  origins = env.scene.env_origins[env_ids]
  robot_yaw = torch.empty(num_envs, device=env.device).uniform_(*robot_yaw_range)
  robot_xy_noise = torch.empty(num_envs, 2, device=env.device).uniform_(
    *robot_xy_noise_range
  )
  robot_xy = origins[:, :2] + robot_xy_noise

  robot_root_state = robot.data.default_root_state[env_ids].clone()
  robot_root_state[:, :2] = robot_xy
  robot_root_state[:, 2] += origins[:, 2]
  zeros = torch.zeros_like(robot_yaw)
  robot_root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, robot_yaw)
  robot_root_state[:, 7:] = 0.0
  robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)

  ball_forward = torch.empty(num_envs, device=env.device).uniform_(*ball_forward_range)
  ball_lateral = torch.empty(num_envs, device=env.device).uniform_(*ball_lateral_range)
  cos_yaw = torch.cos(robot_yaw)
  sin_yaw = torch.sin(robot_yaw)
  ball_xy = torch.empty(num_envs, 2, device=env.device)
  ball_xy[:, 0] = robot_xy[:, 0] + cos_yaw * ball_forward - sin_yaw * ball_lateral
  ball_xy[:, 1] = robot_xy[:, 1] + sin_yaw * ball_forward + cos_yaw * ball_lateral

  ball_root_state = ball.data.default_root_state[env_ids].clone()
  ball_root_state[:, :2] = ball_xy
  ball_root_state[:, 2] = origins[:, 2] + ball_radius
  ball_root_state[:, 3:7] = 0.0
  ball_root_state[:, 3] = 1.0
  ball_root_state[:, 7:9] = torch.empty(num_envs, 2, device=env.device).uniform_(
    *ball_velocity_range
  )
  ball_root_state[:, 9:] = 0.0
  ball.write_root_state_to_sim(ball_root_state, env_ids=env_ids)
