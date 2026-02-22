"""Useful methods for MDP events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def randomize_terrain(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Randomize the sub-terrain for each environment on reset.

  This picks a random terrain type (column) and difficulty level (row) for each
  environment. Useful for play/evaluation mode to test on varied terrains.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  terrain = env.scene.terrain
  if terrain is not None:
    terrain.randomize_env_origins(env_ids)


def reset_scene_to_default(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
  """Reset all entities in the scene to their default states.

  For floating-base entities: Resets root state (position, orientation, velocities).
  For fixed-base mocap entities: Resets mocap pose.
  For all articulated entities: Resets joint positions and velocities.

  Automatically applies env_origins offset to position all entities correctly.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  for entity in env.scene.entities.values():
    if not isinstance(entity, Entity):
      continue

    # Reset root/mocap pose.
    if entity.is_fixed_base and entity.is_mocap:
      # Fixed-base mocap entity - reset mocap pose with env_origins.
      default_root_state = entity.data.default_root_state[env_ids].clone()
      mocap_pose = torch.zeros((len(env_ids), 7), device=env.device)
      mocap_pose[:, 0:3] = default_root_state[:, 0:3] + env.scene.env_origins[env_ids]
      mocap_pose[:, 3:7] = default_root_state[:, 3:7]
      entity.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)
    elif not entity.is_fixed_base:
      # Floating-base entity - reset root state with env_origins.
      default_root_state = entity.data.default_root_state[env_ids].clone()
      default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
      entity.write_root_state_to_sim(default_root_state, env_ids=env_ids)

    # Reset joint state for articulated entities.
    if entity.is_articulated:
      default_joint_pos = entity.data.default_joint_pos[env_ids].clone()
      default_joint_vel = entity.data.default_joint_vel[env_ids].clone()
      entity.write_joint_state_to_sim(
        default_joint_pos, default_joint_vel, env_ids=env_ids
      )


def reset_root_state_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset root state for floating-base or mocap fixed-base entities.

  For floating-base entities: Resets pose and velocity via write_root_state_to_sim().
  For fixed-base mocap entities: Resets pose only via write_mocap_pose_to_sim().

  .. note::
    This function applies the env_origins offset to position entities in a grid.
    For fixed-base robots, this is the ONLY way to position them per-environment.
    Without calling this function in a reset event, fixed-base robots will stack
    at (0,0,0).

  See FAQ: "Why are my fixed-base robots all stacked at the origin?"

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    pose_range: Dictionary with keys {"x", "y", "z", "roll", "pitch", "yaw"}.
    velocity_range: Velocity range (only used for floating-base entities).
    asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]

  # Pose.
  range_list = [
    pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  pose_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )

  # Fixed-based entities with mocap=True.
  if asset.is_fixed_base:
    if not asset.is_mocap:
      raise ValueError(
        f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
      )

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    positions = (
      root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
    )
    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    asset.write_mocap_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    return

  # Floating-base entities.
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  positions = (
    root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
  )
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  # Velocities.
  if velocity_range is None:
    velocity_range = {}
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )

  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_from_flat_patches(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  patch_name: str = "spawn",
  pose_range: dict[str, tuple[float, float]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset root state by placing the asset on a randomly chosen flat patch.

  Selects a random flat patch from the terrain for each environment and positions
  the asset there. Falls back to ``reset_root_state_uniform`` if the terrain has
  no flat patches.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    patch_name: Key into ``terrain.flat_patches`` to use.
    pose_range: Optional random offset applied on top of the patch position.
      Keys: ``{"x", "y", "z", "roll", "pitch", "yaw"}``.
    velocity_range: Optional velocity range (floating-base only).
    asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  terrain = env.scene.terrain
  if terrain is None or patch_name not in terrain.flat_patches:
    reset_root_state_uniform(
      env,
      env_ids,
      pose_range=pose_range or {},
      velocity_range=velocity_range,
      asset_cfg=asset_cfg,
    )
    return

  patches = terrain.flat_patches[patch_name]  # (num_rows, num_cols, num_patches, 3)
  num_patches = patches.shape[2]

  # Look up terrain level (row) and type (col) for each env.
  levels = terrain.terrain_levels[env_ids]
  types = terrain.terrain_types[env_ids]

  # Randomly select a patch index for each env.
  patch_ids = torch.randint(0, num_patches, (len(env_ids),), device=env.device)
  positions = patches[levels, types, patch_ids]

  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  # Apply optional pose range offset.
  if pose_range is None:
    pose_range = {}
  range_list = [
    pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  pose_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )

  # Position: flat patch position + optional offset. Use patch z instead of default.
  final_positions = positions.clone()
  final_positions[:, 0] += pose_samples[:, 0]
  final_positions[:, 1] += pose_samples[:, 1]
  final_positions[:, 2] += root_states[:, 2] + pose_samples[:, 2]

  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  if asset.is_fixed_base:
    if not asset.is_mocap:
      raise ValueError(
        f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
      )
    asset.write_mocap_pose_to_sim(
      torch.cat([final_positions, orientations], dim=-1), env_ids=env_ids
    )
    return

  # Velocities.
  if velocity_range is None:
    velocity_range = {}
  vel_range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  vel_ranges = torch.tensor(vel_range_list, device=env.device)
  vel_samples = sample_uniform(
    vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([final_positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


def push_by_setting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def apply_external_force_torque(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  force_range: tuple[float, float],
  torque_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  asset: Entity = env.scene[asset_cfg.name]
  num_bodies = (
    len(asset_cfg.body_ids)
    if isinstance(asset_cfg.body_ids, list)
    else asset.num_bodies
  )
  size = (len(env_ids), num_bodies, 3)
  forces = sample_uniform(*force_range, size, env.device)
  torques = sample_uniform(*torque_range, size, env.device)
  asset.write_external_wrench_to_sim(
    forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
  )
