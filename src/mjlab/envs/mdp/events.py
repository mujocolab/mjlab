"""Useful methods for MDP events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch

from mjlab.entity import Entity, EntityIndexing
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
  sample_gaussian,
  sample_log_uniform,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  for entity in env.scene.entities.values():
    if not isinstance(entity, Entity):
      continue

    default_root_state = entity.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    entity.write_root_state_to_sim(default_root_state, env_ids=env_ids)

    default_joint_pos = entity.data.default_joint_pos[env_ids].clone()
    default_joint_vel = entity.data.default_joint_vel[env_ids].clone()
    entity.write_joint_state_to_sim(
      default_joint_pos, default_joint_vel, env_ids=env_ids
    )


def reset_root_state_uniform(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset root state for floating-base or mocap fixed-base entities.

  For floating-base entities: Resets pose and velocity via write_root_state_to_sim().
  For fixed-base mocap entities: Resets pose only via write_mocap_pose_to_sim().

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

  velocities[:, 3:] = quat_apply_inverse(orientations, velocities[:, 3:])
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_offset(
  env: ManagerBasedEnv,
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
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  quat_w = asset.data.root_link_quat_w[env_ids]
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
  vel_w[:, 3:] = quat_apply_inverse(quat_w, vel_w[:, 3:])
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def apply_external_force_torque(
  env: ManagerBasedEnv,
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


##
# Domain randomization
##

# TODO: https://github.com/mujocolab/mjlab/issues/38


@dataclass
class FieldSpec:
  """Specification for how to handle a particular field."""

  entity_type: Literal["dof", "joint", "body", "geom", "site", "actuator"]
  use_address: bool = False  # True for fields that need address (q_adr, v_adr)
  default_axes: Optional[List[int]] = None
  valid_axes: Optional[List[int]] = None


FIELD_SPECS = {
  # Dof - uses addresses.
  "dof_armature": FieldSpec("dof", use_address=True),
  "dof_frictionloss": FieldSpec("dof", use_address=True),
  "dof_damping": FieldSpec("dof", use_address=True),
  # Joint - uses IDs directly.
  "jnt_range": FieldSpec("joint"),
  "jnt_stiffness": FieldSpec("joint"),
  # Body - uses IDs directly.
  "body_mass": FieldSpec("body"),
  "body_ipos": FieldSpec("body", default_axes=[0, 1, 2]),
  "body_iquat": FieldSpec("body", default_axes=[0, 1, 2, 3]),
  "body_inertia": FieldSpec("body"),
  "body_pos": FieldSpec("body", default_axes=[0, 1, 2]),
  "body_quat": FieldSpec("body", default_axes=[0, 1, 2, 3]),
  # Geom - uses IDs directly.
  "geom_friction": FieldSpec("geom", default_axes=[0], valid_axes=[0, 1, 2]),
  "geom_pos": FieldSpec("geom", default_axes=[0, 1, 2]),
  "geom_quat": FieldSpec("geom", default_axes=[0, 1, 2, 3]),
  "geom_rgba": FieldSpec("geom", default_axes=[0, 1, 2, 3]),
  # Site - uses IDs directly.
  "site_pos": FieldSpec("site", default_axes=[0, 1, 2]),
  "site_quat": FieldSpec("site", default_axes=[0, 1, 2, 3]),
  # Special case - uses address.
  "qpos0": FieldSpec("joint", use_address=True),
}


def randomize_field(
  env: "ManagerBasedEnv",
  env_ids: torch.Tensor | None,
  field: str,
  ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]],
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  asset_cfg=None,
  axes: Optional[List[int]] = None,
):
  """Unified model randomization function.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize.
    field: Field name (e.g., "geom_friction", "body_mass").
    ranges: Either (min, max) for all axes, or {axis: (min, max)} for specific axes.
    distribution: Distribution type.
    operation: How to apply randomization.
    asset_cfg: Asset configuration.
    axes: Specific axes to randomize (overrides default_axes from field spec).
  """
  if field not in FIELD_SPECS:
    raise ValueError(
      f"Unknown field '{field}'. Supported fields: {list(FIELD_SPECS.keys())}"
    )

  spec = FIELD_SPECS[field]
  asset_cfg = asset_cfg or _DEFAULT_ASSET_CFG
  asset = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  model_field = getattr(env.sim.model, field)

  entity_indices = _get_entity_indices(asset.indexing, asset_cfg, spec)

  target_axes = _determine_target_axes(model_field, spec, axes, ranges)

  axis_ranges = _prepare_axis_ranges(ranges, target_axes, field)

  env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
  indexed_data = model_field[env_grid, entity_grid]

  random_values = _generate_random_values(
    distribution, axis_ranges, indexed_data, target_axes, env.device
  )

  _apply_operation(
    model_field, env_grid, entity_grid, indexed_data, random_values, operation
  )


def _get_entity_indices(
  indexing: EntityIndexing, asset_cfg, spec: FieldSpec
) -> torch.Tensor:
  match spec.entity_type:
    case "dof":
      return indexing.joint_v_adr[asset_cfg.joint_ids]
    case "joint" if spec.use_address:
      return indexing.joint_q_adr[asset_cfg.joint_ids]
    case "joint":
      return indexing.joint_ids[asset_cfg.joint_ids]
    case "body":
      return indexing.body_ids[asset_cfg.body_ids]
    case "geom":
      return indexing.geom_ids[asset_cfg.geom_ids]
    case "site":
      return indexing.site_ids[asset_cfg.site_ids]
    case "actuator":
      assert indexing.ctrl_ids is not None
      return indexing.ctrl_ids[asset_cfg.actuator_ids]
    case _:
      raise ValueError(f"Unknown entity type: {spec.entity_type}")


def _determine_target_axes(
  model_field,
  spec: FieldSpec,
  axes: Optional[List[int]],
  ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]],
) -> List[int]:
  """Determine which axes to randomize."""
  field_ndim = len(model_field.shape) - 1  # Subtract env dimension

  if axes is not None:
    # User specified axes explicitly.
    target_axes = axes
  elif isinstance(ranges, dict):
    # Axes specified via dictionary keys.
    target_axes = list(ranges.keys())
  elif spec.default_axes is not None:
    # Use field specification defaults.
    target_axes = spec.default_axes
  else:
    # Randomize all axes.
    if field_ndim > 1:
      target_axes = list(range(model_field.shape[-1]))  # Last dimension
    else:
      target_axes = [0]  # Scalar field.

  # Validate axes
  if spec.valid_axes is not None:
    invalid_axes = set(target_axes) - set(spec.valid_axes)
    if invalid_axes:
      raise ValueError(
        f"Invalid axes {invalid_axes} for field. Valid axes: {spec.valid_axes}"
      )

  return target_axes


def _prepare_axis_ranges(
  ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]],
  target_axes: List[int],
  field: str,
) -> Dict[int, Tuple[float, float]]:
  """Convert ranges to a consistent dictionary format."""
  if isinstance(ranges, tuple):
    # Same range for all axes.
    return {axis: ranges for axis in target_axes}
  elif isinstance(ranges, dict):
    # Validate that all target axes have ranges.
    missing_axes = set(target_axes) - set(ranges.keys())
    if missing_axes:
      raise ValueError(
        f"Missing ranges for axes {missing_axes} in field '{field}'. "
        f"Required axes: {target_axes}"
      )
    return {axis: ranges[axis] for axis in target_axes}
  else:
    raise TypeError(f"ranges must be tuple or dict, got {type(ranges)}")


def _generate_random_values(
  distribution: str,
  axis_ranges: Dict[int, Tuple[float, float]],
  indexed_data: torch.Tensor,
  target_axes: List[int],
  device,
) -> torch.Tensor:
  """Generate random values for the specified axes."""
  result = indexed_data.clone()

  for axis in target_axes:
    lower, upper = axis_ranges[axis]
    lower_bound = torch.tensor([lower], device=device)
    upper_bound = torch.tensor([upper], device=device)

    if len(indexed_data.shape) > 2:  # Multi-dimensional field.
      shape = (*indexed_data.shape[:-1], 1)  # Same shape but single axis.
    else:
      shape = indexed_data.shape

    random_vals = _sample_distribution(
      distribution, lower_bound, upper_bound, shape, device
    )

    if len(indexed_data.shape) > 2:
      result[..., axis] = random_vals.squeeze(-1)
    else:
      result = random_vals

  return result


def _apply_operation(
  model_field,
  env_grid,
  entity_grid,
  indexed_data,
  random_values,
  operation,
):
  """Apply the randomization operation."""
  if operation == "add":
    model_field[env_grid, entity_grid] = indexed_data + random_values
  elif operation == "scale":
    model_field[env_grid, entity_grid] = indexed_data * random_values
  elif operation == "abs":
    model_field[env_grid, entity_grid] = random_values
  else:
    raise ValueError(f"Unknown operation: {operation}")


def _sample_distribution(
  distribution: str,
  lower: torch.Tensor,
  upper: torch.Tensor,
  shape: tuple,
  device: str,
) -> torch.Tensor:
  """Sample from the specified distribution."""
  if distribution == "uniform":
    return sample_uniform(lower, upper, shape, device=device)
  elif distribution == "log_uniform":
    return sample_log_uniform(lower, upper, shape, device=device)
  elif distribution == "gaussian":
    return sample_gaussian(lower, upper, shape, device=device)
  else:
    raise ValueError(f"Unknown distribution: {distribution}")
