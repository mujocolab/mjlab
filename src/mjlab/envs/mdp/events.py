"""Useful methods for MDP events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Tuple

import torch

from mjlab.entity import Entity, EntityIndexing
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_gaussian,
  sample_log_uniform,
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


##
# Domain randomization
##

# TODO: https://github.com/mujocolab/mjlab/issues/38


@dataclass
class FieldSpec:
  """Specification for how to handle a particular field."""

  entity_type: Literal["dof", "joint", "body", "geom", "site", "actuator"]
  use_address: bool = False  # True for fields that need address (q_adr, v_adr)
  default_axes: list[int] | None = None
  valid_axes: list[int] | None = None


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
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  field: str,
  ranges: Tuple[float, float] | Dict[int, Tuple[float, float]],
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  asset_cfg=None,
  axes: list[int] | None = None,
  shared_random: bool = False,
):
  """Unified model randomization function.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize.
    field: Field name (e.g., "geom_friction", "body_mass").
    ranges: Either (min, max) for all axes, or {axis: (min, max)} for specific axes.
    distribution: Distribution type.
    operation: How to apply randomization. For "scale" and "add" operations,
      values are computed from stored defaults (not current values) to prevent
      accumulation when randomizing on reset.
    asset_cfg: Asset configuration.
    axes: Specific axes to randomize (overrides default_axes from field spec).
    shared_random: If True, sample a single random value per environment and apply
      it to all entities. Useful when multiple geometries should share the same
      randomized value (e.g., all foot collision geoms sharing friction).
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

  # For scale/add operations, use stored default values to prevent accumulation.
  if operation in ("scale", "add"):
    default_field = env.sim.get_default_field(field)
    base_values = default_field[entity_indices].unsqueeze(0).expand_as(indexed_data)
  else:
    base_values = indexed_data

  if shared_random:
    # Generate a single random value per environment and broadcast to all entities.
    single_entity_values = base_values[:, :1]  # (num_envs, 1, ...)
    random_values = _generate_random_values(
      distribution,
      axis_ranges,
      single_entity_values,
      target_axes,
      env.device,
      operation,
    )
    random_values = random_values.expand_as(base_values)
  else:
    random_values = _generate_random_values(
      distribution, axis_ranges, base_values, target_axes, env.device, operation
    )

  _apply_operation(
    model_field, env_grid, entity_grid, base_values, random_values, operation
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
  axes: list[int] | None,
  ranges: Tuple[float, float] | Dict[int, Tuple[float, float]],
) -> list[int]:
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
  ranges: Tuple[float, float] | Dict[int, Tuple[float, float]],
  target_axes: list[int],
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
  target_axes: list[int],
  device,
  operation: str,
) -> torch.Tensor:
  """Generate random values for the specified axes.

  For scale/add operations, non-randomized axes use identity values (1.0 for
  scale, 0.0 for add) to prevent modification. For abs operations, non-randomized
  axes preserve their current values.
  """
  if operation == "scale":
    result = torch.ones_like(indexed_data)
  elif operation == "add":
    result = torch.zeros_like(indexed_data)
  else:
    assert operation == "abs"
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


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: Tuple[float, float],
  kd_range: Tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize PD stiffness and damping gains.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all environments.
    kp_range: (min, max) for proportional gain randomization.
    kd_range: (min, max) for derivative gain randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default gains by sampled values, "abs" sets absolute
      values. For "scale" operations, values are computed from stored defaults to
      prevent accumulation when randomizing on reset.
  """
  from mjlab.actuator import (
    BuiltinPositionActuator,
    IdealPdActuator,
    XmlPositionActuator,
  )
  from mjlab.actuator.delayed_actuator import DelayedActuator

  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  elif isinstance(asset_cfg.actuator_ids, slice):
    actuators = asset.actuators[asset_cfg.actuator_ids]
  else:
    actuators = [asset.actuators[asset_cfg.actuator_ids]]

  # Unwrap DelayedActuators to access base actuators.
  actuators = [
    a.base_actuator if isinstance(a, DelayedActuator) else a for a in actuators
  ]

  for actuator in actuators:
    ctrl_ids = actuator.ctrl_ids

    kp_samples = _sample_distribution(
      distribution,
      torch.tensor(kp_range[0], device=env.device),
      torch.tensor(kp_range[1], device=env.device),
      (len(env_ids), len(ctrl_ids)),
      env.device,
    )
    kd_samples = _sample_distribution(
      distribution,
      torch.tensor(kd_range[0], device=env.device),
      torch.tensor(kd_range[1], device=env.device),
      (len(env_ids), len(ctrl_ids)),
      env.device,
    )

    if isinstance(actuator, (BuiltinPositionActuator, XmlPositionActuator)):
      if operation == "scale":
        # Use stored default values for scaling to prevent accumulation.
        default_gainprm = env.sim.get_default_field("actuator_gainprm")
        default_biasprm = env.sim.get_default_field("actuator_biasprm")
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
          default_gainprm[ctrl_ids, 0] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
          default_biasprm[ctrl_ids, 1] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
          default_biasprm[ctrl_ids, 2] * kd_samples
        )
      elif operation == "abs":
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = -kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = -kd_samples

    elif isinstance(actuator, IdealPdActuator):
      assert actuator.stiffness is not None
      assert actuator.damping is not None
      if operation == "scale":
        # Use default gains for scaling to prevent accumulation.
        assert actuator.default_stiffness is not None
        assert actuator.default_damping is not None
        actuator.set_gains(
          env_ids,
          kp=actuator.default_stiffness[env_ids] * kp_samples,
          kd=actuator.default_damping[env_ids] * kd_samples,
        )
      elif operation == "abs":
        actuator.set_gains(env_ids, kp=kp_samples, kd=kd_samples)

    else:
      raise TypeError(
        f"randomize_pd_gains only supports BuiltinPositionActuator, XmlPositionActuator, "
        f"and IdealPdActuator (optionally wrapped with DelayedActuator), "
        f"got {type(actuator).__name__}"
      )


@requires_model_fields("actuator_forcerange")
def randomize_effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  effort_limit_range: Tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize actuator effort limits.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all environments.
    effort_limit_range: (min, max) for effort limit randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies existing limits, "abs" sets absolute values.
  """
  from mjlab.actuator import (
    BuiltinPositionActuator,
    IdealPdActuator,
    XmlPositionActuator,
  )

  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  else:
    actuators = asset.actuators[asset_cfg.actuator_ids]

  if not isinstance(actuators, list):
    actuators = [actuators]

  for actuator in actuators:
    ctrl_ids = actuator.ctrl_ids
    num_actuators = len(ctrl_ids)

    effort_samples = _sample_distribution(
      distribution,
      torch.tensor(effort_limit_range[0], device=env.device),
      torch.tensor(effort_limit_range[1], device=env.device),
      (len(env_ids), num_actuators),
      env.device,
    )

    if isinstance(actuator, (BuiltinPositionActuator, XmlPositionActuator)):
      if operation == "scale":
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] *= (
          effort_samples
        )
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] *= (
          effort_samples
        )
      elif operation == "abs":
        env.sim.model.actuator_forcerange[
          env_ids[:, None], ctrl_ids, 0
        ] = -effort_samples
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          effort_samples
        )

    elif isinstance(actuator, IdealPdActuator):
      assert actuator.force_limit is not None
      if operation == "scale":
        current_limit = actuator.force_limit[env_ids].clone()
        actuator.set_effort_limit(env_ids, effort_limit=current_limit * effort_samples)
      elif operation == "abs":
        actuator.set_effort_limit(env_ids, effort_limit=effort_samples)

    else:
      raise TypeError(
        f"randomize_effort_limits only supports BuiltinPositionActuator, XmlPositionActuator, "
        f"and IdealPdActuator, got {type(actuator).__name__}"
      )


def randomize_encoder_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  bias_range: Tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize encoder bias to simulate joint encoder calibration errors.

  See docs/api/domain_randomization.md for details on how encoder bias works.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    num_joints = asset.num_joints
    joint_ids_tensor = torch.arange(num_joints, device=env.device)
  else:
    joint_ids_tensor = torch.tensor(joint_ids, device=env.device)

  num_joints = len(joint_ids_tensor)
  bias_samples = sample_uniform(
    torch.tensor(bias_range[0], device=env.device),
    torch.tensor(bias_range[1], device=env.device),
    (len(env_ids), num_joints),
    env.device,
  )

  if isinstance(joint_ids, slice):
    asset.data.encoder_bias[env_ids] = bias_samples
  else:
    asset.data.encoder_bias[env_ids[:, None], joint_ids_tensor] = bias_samples


def sync_actuator_delays(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  lag_range: Tuple[int, int],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Synchronize delay lags across all delayed actuators.

  Samples a single lag value per environment and applies it to all delayed
  actuators. Useful for simulating the same delay across actuator groups.

  Args:
    env: The environment.
    env_ids: Environment IDs to set. If None, sets all environments.
    lag_range: (min_lag, max_lag) range for sampling lag values in physics
      timesteps.
    asset_cfg: Asset configuration specifying which entity and actuators.
  """
  from mjlab.actuator.delayed_actuator import DelayedActuator

  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  elif isinstance(asset_cfg.actuator_ids, slice):
    actuators = asset.actuators[asset_cfg.actuator_ids]
  else:
    actuators = [asset.actuators[asset_cfg.actuator_ids]]

  # Filter to only delayed actuators.
  delayed_actuators = [a for a in actuators if isinstance(a, DelayedActuator)]

  if not delayed_actuators:
    return

  # Sample one lag per environment (shared across all actuators).
  lags = torch.randint(
    lag_range[0],
    lag_range[1] + 1,
    (len(env_ids),),
    device=env.device,
    dtype=torch.long,
  )

  # Apply the same lag to all delayed actuators.
  for actuator in delayed_actuators:
    actuator.set_lags(lags, env_ids)
