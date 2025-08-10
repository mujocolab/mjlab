from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entities.robots.robot import Robot
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  sample_uniform,
  sample_log_uniform,
  sample_gaussian,
  quat_from_euler_xyz,
  quat_apply_inverse,
  quat_mul,
)

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
) -> None:
  asset: Robot = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  # Positions.
  range_list = [
    pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  rand_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )

  positions = root_states[:, 0:3] + rand_samples[:, 0:3]
  orientations_delta = quat_from_euler_xyz(
    rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  # Velocities.
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  rand_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + rand_samples

  asset.write_root_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )

  velocities[:, 3:] = quat_apply_inverse(orientations, velocities[:, 3:])
  asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_scale(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
) -> None:
  asset: Robot = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()

  joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=asset_cfg.joint_ids,
  )


def push_by_setting_velocity(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
) -> None:
  asset: Robot = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  quat_w = asset.data.root_link_quat_w[env_ids]
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
  vel_w[:, 3:] = quat_apply_inverse(quat_w, vel_w[:, 3:])
  asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


##
# Domain randomization
##

"""
Dof: armature, frictionloss, damping, solref/solimp. 
Jnt: limits, solref/solimp.
Site: pos, quat.
Body: pos, quat, inertia, mass.
Geom: solref/solimp, friction, pos, quat, color.
Misc: gravity, etc.
"""


def randomize_model_field(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  field: str,
  distribution_params: dict[str, tuple[float, float]],
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
) -> None:
  asset: Robot = env.scene[asset_cfg.name]

  # Default to all environments if none specified.
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  model_field = getattr(env.sim.model, field)

  if asset_cfg.body_ids == slice(None):
    body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device=env.device)
  else:
    body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device=env.device)

  # Extract bounds for randomization.
  lower_bounds = torch.tensor(
    [v[0] for v in distribution_params.values()], device=env.device
  )
  upper_bounds = torch.tensor(
    [v[1] for v in distribution_params.values()], device=env.device
  )

  env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
  indexed_data = model_field[env_grid, body_grid]

  random_values = _sample_distribution(
    distribution, lower_bounds, upper_bounds, indexed_data.shape, env.device
  )

  if operation == "add":
    model_field[env_grid, body_grid] = indexed_data + random_values
  elif operation == "scale":
    model_field[env_grid, body_grid] = indexed_data * random_values
  elif operation == "abs":
    model_field[env_grid, body_grid] = random_values
  else:
    raise ValueError(f"Unknown operation: {operation}")


def _sample_distribution(
  distribution: str,
  lower: torch.Tensor,
  upper: torch.Tensor,
  shape: tuple,
  device: torch.device,
) -> torch.Tensor:
  if distribution == "uniform":
    return sample_uniform(lower, upper, shape, device=device)
  elif distribution == "log_uniform":
    return sample_log_uniform(lower, upper, shape, device=device)
  elif distribution == "gaussian":
    return sample_gaussian(lower, upper, shape, device=device)
  else:
    raise ValueError(f"Unknown distribution: {distribution}")
