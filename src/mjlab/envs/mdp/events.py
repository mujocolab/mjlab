from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch

from mjlab.entity.robots.robot import Robot
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
  sample_gaussian,
  sample_log_uniform,
  sample_uniform,
)
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
  for entity in env.scene.entities.values():
    if not isinstance(entity, Robot):
      continue

    default_root_state = entity.data.default_root_state[env_ids].clone()
    entity.write_root_state_to_sim(default_root_state, env_ids=env_ids)

    default_joint_pos = entity.data.default_joint_pos[env_ids].clone()
    default_joint_vel = entity.data.default_joint_vel[env_ids].clone()
    entity.write_joint_state_to_sim(
      default_joint_pos, default_joint_vel, env_ids=env_ids
    )


def reset_root_state_uniform(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
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

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )

  velocities[:, 3:] = quat_apply_inverse(orientations, velocities[:, 3:])
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_scale(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
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
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


##
# Domain randomization
##

# TODO: Clean this up and figure out how to support general fields such as
# model.opt.gravity and model.opt.impratio.
# TODO: Need to write other things back to asset.data when changing certain fields such
# as joint limits, armature, etc.


@dataclass
class FieldSpec:
  """Specification for how to handle a particular field."""

  entity_type: Literal["dof", "joint", "body", "geom", "site", "actuator"]
  indexing_attr: str  # e.g., "joint_v_adr", "joint_local2global", etc.
  default_axes: Optional[List[int]] = None  # Which axes to randomize by default
  valid_axes: Optional[List[int]] = None  # Which axes are valid for this field


FIELD_SPECS = {
  # Dof.
  "dof_armature": FieldSpec("dof", "joint_v_adr"),
  "dof_frictionloss": FieldSpec("dof", "joint_v_adr"),
  "dof_damping": FieldSpec("dof", "joint_v_adr"),
  # Joint.
  "jnt_range": FieldSpec("joint", "joint_local2global"),
  "jnt_stiffness": FieldSpec("joint", "joint_local2global"),
  # Body.
  "body_mass": FieldSpec("body", "body_local2global"),
  "body_ipos": FieldSpec("body", "body_local2global", default_axes=[0, 1, 2]),
  "body_iquat": FieldSpec("body", "body_local2global", default_axes=[0, 1, 2, 3]),
  "body_inertia": FieldSpec("body", "body_local2global"),
  "body_pos": FieldSpec("body", "body_local2global", default_axes=[0, 1, 2]),
  "body_quat": FieldSpec("body", "body_local2global", default_axes=[0, 1, 2, 3]),
  # Geom.
  "geom_friction": FieldSpec(
    "geom", "geom_local2global", default_axes=[0], valid_axes=[0, 1, 2]
  ),
  "geom_pos": FieldSpec("geom", "geom_local2global", default_axes=[0, 1, 2]),
  "geom_quat": FieldSpec("geom", "geom_local2global", default_axes=[0, 1, 2, 3]),
  "geom_rgba": FieldSpec("geom", "geom_local2global", default_axes=[0, 1, 2, 3]),
  # Site.
  "site_pos": FieldSpec("site", "site_local2global", default_axes=[0, 1, 2]),
  "site_quat": FieldSpec("site", "site_local2global", default_axes=[0, 1, 2, 3]),
  # Other.
  "qpos0": FieldSpec("joint", "joint_q_adr"),
}


def randomize_field(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
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

  entity_indices = _get_entity_indices(env, asset, asset_cfg, spec)

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


def _get_entity_indices(env, asset, asset_cfg, spec: FieldSpec) -> torch.Tensor:
  """Get the global indices for the entities to randomize."""
  indexing = env.scene.indexing.entities[asset_cfg.name]

  if spec.entity_type == "dof":
    if asset_cfg.joint_ids == slice(None):
      local_ids = list(range(asset.num_joints))
    else:
      local_ids = asset_cfg.joint_ids
    return indexing.joint_v_adr[local_ids]

  elif spec.entity_type == "joint":
    if asset_cfg.joint_ids == slice(None):
      local_ids = list(range(asset.num_joints))
    else:
      local_ids = asset_cfg.joint_ids

    if spec.indexing_attr == "joint_q_adr":  # qpos0.
      return indexing.joint_q_adr[local_ids]
    else:
      return torch.tensor(
        [indexing.joint_local2global[lid] for lid in local_ids],
        dtype=torch.int,
        device=env.device,
      )

  elif spec.entity_type == "body":
    if asset_cfg.body_ids == slice(None):
      local_ids = list(range(asset.num_bodies))
    else:
      local_ids = asset_cfg.body_ids
    return torch.tensor(
      [indexing.body_local2global[lid] for lid in local_ids],
      dtype=torch.int,
      device=env.device,
    )

  elif spec.entity_type == "geom":
    if asset_cfg.geom_ids == slice(None):
      local_ids = list(range(asset.num_geoms))
    else:
      local_ids = asset_cfg.geom_ids
    return torch.tensor(
      [indexing.geom_local2global[lid] for lid in local_ids],
      dtype=torch.int,
      device=env.device,
    )

  elif spec.entity_type == "site":
    if asset_cfg.site_ids == slice(None):
      local_ids = list(range(asset.num_sites))
    else:
      local_ids = asset_cfg.site_ids
    return torch.tensor(
      [indexing.site_local2global[lid] for lid in local_ids],
      dtype=torch.int,
      device=env.device,
    )

  else:
    raise ValueError(f"Unknown entity type: {spec.entity_type}")


def _determine_target_axes(
  model_field,
  spec: FieldSpec,
  axes: Optional[List[int]],
  ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]],
) -> List[int]:
  """Determine which axes to randomize."""
  field_ndim = len(model_field.shape) - 1  # Subtract env dimension.

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
      target_axes = list(range(model_field.shape[-1]))  # Last dimension.
    else:
      target_axes = [0]  # Scalar field.

  # Validate axes.
  if spec.valid_axes is not None:
    invalid_axes = set(target_axes) - set(spec.valid_axes)
    if invalid_axes:
      raise ValueError(
        f"Invalid axes {invalid_axes} for field '{model_field.name}'. "
        f"Valid axes: {spec.valid_axes}"
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


def randomize_actuator_gains(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  stiffness_distribution_params: tuple[float, float] | None = None,
  damping_distribution_params: tuple[float, float] | None = None,
  operation: Literal["add", "scale", "abs"] = "abs",
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
  asset: Robot = env.scene[asset_cfg.name]
  indexing = env.scene.indexing.entities[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  local_act_ids = resolve_matching_names(
    asset.actuator_names, asset.joint_actuators, True
  )[0]
  act_ids = torch.tensor(
    [indexing.actuator_local2global[lid] for lid in local_act_ids],
    dtype=torch.int,
    device=env.device,
  )
  local_act_ids = torch.tensor(
    local_act_ids,
    dtype=torch.int,
    device=env.device,
  )

  env_grid, act_grid = torch.meshgrid(env_ids, act_ids, indexing="ij")

  if stiffness_distribution_params is not None:
    stiffness_field = env.sim.model.actuator_gainprm
    stiffness_lower, stiffness_upper = stiffness_distribution_params

    current_stiffness = stiffness_field[env_grid, act_grid, 0]

    stiffness_random_values = _sample_distribution(
      distribution,
      torch.tensor([stiffness_lower], device=env.device),
      torch.tensor([stiffness_upper], device=env.device),
      current_stiffness.shape,
      env.device,
    )

    if operation == "add":
      new_val = current_stiffness + stiffness_random_values
    elif operation == "scale":
      new_val = current_stiffness * stiffness_random_values
    elif operation == "abs":
      new_val = stiffness_random_values
    else:
      raise ValueError(f"Unknown operation: {operation}")

    env.sim.model.actuator_gainprm[env_grid, act_grid, 0] = new_val
    env.sim.model.actuator_biasprm[env_grid, act_grid, 1] = -new_val
    asset.write_joint_stiffness_to_sim(
      stiffness=new_val,
      joint_ids=local_act_ids,
      env_ids=env_ids,
    )

  if damping_distribution_params is not None:
    damping_field = env.sim.model.actuator_biasprm
    damping_lower, damping_upper = damping_distribution_params

    current_damping = damping_field[env_grid, act_grid, 1]

    damping_random_values = _sample_distribution(
      distribution,
      torch.tensor([damping_lower], device=env.device),
      torch.tensor([damping_upper], device=env.device),
      current_damping.shape,
      env.device,
    )

    if operation == "add":
      new_val = current_damping + damping_random_values
    elif operation == "scale":
      new_val = current_damping * damping_random_values
    elif operation == "abs":
      new_val = damping_random_values
    else:
      raise ValueError(f"Unknown operation: {operation}")

    env.sim.model.actuator_biasprm[env_grid, act_grid, 1] = -new_val
    asset.write_joint_damping_to_sim(
      damping=new_val,
      joint_ids=local_act_ids,
      env_ids=env_ids,
    )
