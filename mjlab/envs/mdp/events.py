from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.utils import math as math_utils
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entities.robots.robot import Robot

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
  raise NotImplementedError


def reset_root_state_uniform(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
):
  asset: Robot = env.scene.entities[asset_cfg.name]
  root_states = asset.data.default_root_state[env_ids].clone()

  # Positions.
  range_list = [
    pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  rand_samples = math_utils.sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )

  positions = root_states[:, 0:3] + rand_samples[:, 0:3]
  orientations_delta = math_utils.quat_from_euler_xyz(
    rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
  )
  orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

  # Velocities.
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  rand_samples = math_utils.sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + rand_samples

  asset.write_root_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_scale(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
):
  asset: Robot = env.scene[asset_cfg.name]
  # joint_pos = asset.data.default_joint_pos[env_ids, asset_cfg.joint_ids].clone()
  joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  # joint_vel = asset.data.default_joint_vel[env_ids, asset_cfg.joint_ids].clone()
  joint_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()

  joint_pos *= math_utils.sample_uniform(
    *position_range, joint_pos.shape, joint_pos.device
  )
  joint_vel *= math_utils.sample_uniform(
    *velocity_range, joint_vel.shape, joint_vel.device
  )

  joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=asset_cfg.joint_ids,
  )
