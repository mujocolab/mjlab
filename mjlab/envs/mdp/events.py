from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.utils import random as rand_utils
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
  print("hi")


def reset_root_state_uniform(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
):
  asset = env.scene.entities[asset_cfg.name]
  # root_states = asset.data._default_root_state[env_ids].clone()
  root_states = torch.tensor(asset._default_root_state, device=env.device)[None].repeat(
    env.num_envs, 1
  )

  range_list = [
    pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  rand_samples = rand_utils.sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )

  positions = root_states[:, 0:3] + rand_samples[:, 0:3]
  orientations_delta = rand_utils.quat_from_euler_xyz(
    rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
  )
  orientations = rand_utils.quat_mul(root_states[:, 3:7], orientations_delta)
  # velocities
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  rand_samples = rand_utils.sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + rand_samples

  env.sim.data.qpos[:, :3] = positions
  env.sim.data.qpos[:, 3:7] = orientations
  env.sim.data.qpos[:, 7:13] = velocities


def reset_joints_by_scale(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
):
  asset = env.scene.entities[asset_cfg.name]
  joint_pos = torch.tensor(asset._default_joint_pos, device=env.device)[None].repeat(
    env.num_envs, 1
  )
  joint_vel = torch.tensor(asset._default_joint_vel, device=env.device)[None].repeat(
    env.num_envs, 1
  )

  joint_pos *= rand_utils.sample_uniform(
    *position_range, joint_pos.shape, joint_pos.device
  )
  joint_vel *= rand_utils.sample_uniform(
    *velocity_range, joint_vel.shape, joint_vel.device
  )

  joint_pos_limits = env.sim.model.jnt_range[:, 1:]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  env.sim.data.qpos[:, asset_cfg.qpos_ids[7:]] = joint_pos
  env.sim.data.qvel[:, asset_cfg.dof_ids[6:]] = joint_vel
