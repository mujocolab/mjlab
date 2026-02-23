"""Shared private engine for domain randomization."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity, EntityIndexing
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from ._types import (
  Distribution,
  Operation,
  resolve_distribution,
  resolve_operation,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

Ranges = (
  tuple[float, float] | dict[int, tuple[float, float]] | dict[str, tuple[float, float]]
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Mapping from entity_type to the names attribute on Entity / SceneEntityCfg.
_ENTITY_NAMES_ATTR: dict[str, str] = {
  "dof": "joint_names",
  "joint": "joint_names",
  "body": "body_names",
  "geom": "geom_names",
  "site": "site_names",
  "actuator": "actuator_names",
  "tendon": "tendon_names",
  "camera": "camera_names",
  "light": "light_names",
}

# Private engine.


def _randomize_model_field(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  field: str,
  *,
  entity_type: str,
  ranges: Ranges,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  axes: list[int] | None = None,
  shared_random: bool = False,
  default_axes: list[int] | None = None,
  valid_axes: list[int] | None = None,
  use_address: bool = False,
) -> None:
  """Core randomization engine for model fields."""
  operation = resolve_operation(operation)
  distribution = resolve_distribution(distribution)

  # Handle string-keyed ranges: resolve each pattern and recurse.
  if isinstance(ranges, dict) and ranges and isinstance(next(iter(ranges)), str):
    _randomize_with_string_ranges(
      env,
      env_ids,
      field,
      entity_type=entity_type,
      ranges=ranges,  # type: ignore[arg-type]
      distribution=distribution,
      operation=operation,
      asset_cfg=asset_cfg,
      axes=axes,
      shared_random=shared_random,
      default_axes=default_axes,
      valid_axes=valid_axes,
      use_address=use_address,
    )
    return

  # At this point, string-keyed ranges have been handled above.
  narrowed_ranges: tuple[float, float] | dict[int, tuple[float, float]] = ranges  # type: ignore[assignment]

  asset = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  model_field = getattr(env.sim.model, field)
  entity_indices = _get_entity_indices(
    asset.indexing, asset_cfg, entity_type, use_address
  )
  target_axes = _determine_target_axes(
    model_field, axes, narrowed_ranges, default_axes, valid_axes
  )
  axis_ranges = _prepare_axis_ranges(narrowed_ranges, target_axes, field)

  env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
  indexed_data = model_field[env_grid, entity_grid]

  if operation.uses_defaults:
    default_field = env.sim.get_default_field(field)
    base_values = default_field[entity_indices].unsqueeze(0).expand_as(indexed_data)
  else:
    base_values = indexed_data

  if shared_random:
    single_entity_values = base_values[:, :1]
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
      distribution,
      axis_ranges,
      base_values,
      target_axes,
      env.device,
      operation,
    )

  model_field[env_grid, entity_grid] = operation.combine(base_values, random_values)


def _randomize_with_string_ranges(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  field: str,
  *,
  entity_type: str,
  ranges: dict[str, tuple[float, float]],
  distribution: Distribution | str,
  operation: Operation | str,
  asset_cfg: SceneEntityCfg,
  axes: list[int] | None,
  shared_random: bool,
  default_axes: list[int] | None,
  valid_axes: list[int] | None,
  use_address: bool,
) -> None:
  """Resolve string-keyed ranges to entity names and recurse."""
  asset: Entity = env.scene[asset_cfg.name]
  names_attr = _ENTITY_NAMES_ATTR[entity_type]
  all_names = list(getattr(asset, names_attr))

  for pattern, range_val in ranges.items():
    matched = [n for n in all_names if re.fullmatch(pattern, n)]
    if not matched:
      raise ValueError(
        f"Pattern '{pattern}' matched no {entity_type} names in "
        f"entity '{asset_cfg.name}'. Available: {all_names}"
      )
    sub_cfg = SceneEntityCfg(asset_cfg.name)
    setattr(sub_cfg, names_attr, tuple(matched))
    sub_cfg.resolve(env.scene)
    _randomize_model_field(
      env,
      env_ids,
      field,
      entity_type=entity_type,
      ranges=range_val,
      distribution=distribution,
      operation=operation,
      asset_cfg=sub_cfg,
      axes=axes,
      shared_random=shared_random,
      default_axes=default_axes,
      valid_axes=valid_axes,
      use_address=use_address,
    )


def _get_entity_indices(
  indexing: EntityIndexing,
  asset_cfg: SceneEntityCfg,
  entity_type: str,
  use_address: bool,
) -> torch.Tensor:
  match entity_type:
    case "dof":
      return indexing.joint_v_adr[asset_cfg.joint_ids]
    case "joint" if use_address:
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
    case "tendon":
      return indexing.tendon_ids[asset_cfg.tendon_ids]
    case "camera":
      return indexing.cam_ids[asset_cfg.camera_ids]
    case "light":
      return indexing.light_ids[asset_cfg.light_ids]
    case _:
      raise ValueError(f"Unknown entity type: {entity_type}")


def _determine_target_axes(
  model_field: torch.Tensor,
  axes: list[int] | None,
  ranges: Ranges,
  default_axes: list[int] | None,
  valid_axes: list[int] | None,
) -> list[int]:
  """Determine which axes to randomize."""
  field_ndim = len(model_field.shape) - 1

  if axes is not None:
    target_axes = axes
  elif isinstance(ranges, dict) and ranges and isinstance(next(iter(ranges)), int):
    target_axes = [k for k in ranges if isinstance(k, int)]
  elif default_axes is not None:
    target_axes = default_axes
  else:
    if field_ndim > 1:
      target_axes = list(range(model_field.shape[-1]))
    else:
      target_axes = [0]

  if valid_axes is not None:
    invalid_axes = set(target_axes) - set(valid_axes)
    if invalid_axes:
      raise ValueError(
        f"Invalid axes {invalid_axes} for field. Valid axes: {valid_axes}"
      )

  return target_axes


def _prepare_axis_ranges(
  ranges: tuple[float, float] | dict[int, tuple[float, float]],
  target_axes: list[int],
  field: str,
) -> dict[int, tuple[float, float]]:
  """Convert ranges to a consistent dictionary format."""
  if isinstance(ranges, tuple):
    return {axis: ranges for axis in target_axes}
  missing_axes = set(target_axes) - set(ranges.keys())
  if missing_axes:
    raise ValueError(
      f"Missing ranges for axes {missing_axes} in field '{field}'. "
      f"Required axes: {target_axes}"
    )
  return {axis: ranges[axis] for axis in target_axes}


def _generate_random_values(
  distribution: Distribution,
  axis_ranges: dict[int, tuple[float, float]],
  indexed_data: torch.Tensor,
  target_axes: list[int],
  device: str,
  operation: Operation,
) -> torch.Tensor:
  """Generate random values for the specified axes."""
  result = operation.initialize(indexed_data)

  for axis in target_axes:
    lower, upper = axis_ranges[axis]
    lower_bound = torch.tensor([lower], device=device)
    upper_bound = torch.tensor([upper], device=device)

    if len(indexed_data.shape) > 2:
      shape = (*indexed_data.shape[:-1], 1)
    else:
      shape = indexed_data.shape

    random_vals = distribution.sample(lower_bound, upper_bound, shape, device)

    if len(indexed_data.shape) > 2:
      result[..., axis] = random_vals.squeeze(-1)
    else:
      result = random_vals

  return result


# Quaternion helpers.


def _sample_angle(
  distribution: Distribution | str,
  range_: tuple[float, float],
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  dist = resolve_distribution(distribution)
  lower = torch.tensor(range_[0], device=device)
  upper = torch.tensor(range_[1], device=device)
  return dist.sample(lower, upper, shape, device)


def _randomize_quat_field(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  field: str,
  *,
  entity_type: str,
  roll_range: tuple[float, float],
  pitch_range: tuple[float, float],
  yaw_range: tuple[float, float],
  distribution: Distribution | str,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Core implementation for quaternion randomization via RPY composition.

  Composes a sampled RPY perturbation with the default quaternion to
  produce a valid unit quaternion. Axes with range ``(0.0, 0.0)`` sample
  zero and leave that rotation component unchanged.
  """
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  entity_indices = _get_entity_indices(asset.indexing, asset_cfg, entity_type, False)
  n_envs = len(env_ids)
  n_entities = len(entity_indices)
  shape = (n_envs, n_entities)

  roll = _sample_angle(distribution, roll_range, shape, env.device)
  pitch = _sample_angle(distribution, pitch_range, shape, env.device)
  yaw = _sample_angle(distribution, yaw_range, shape, env.device)

  # quat_from_euler_xyz requires 1-D inputs; reshape back after.
  q_perturb = quat_from_euler_xyz(
    roll.flatten(), pitch.flatten(), yaw.flatten()
  ).reshape(n_envs, n_entities, 4)

  # Expand default to (n_envs, n_entities, 4) so quat_mul shapes match.
  q_default = env.sim.get_default_field(field)[entity_indices]  # (n_entities, 4)
  q_default_exp = q_default.unsqueeze(0).expand(n_envs, n_entities, 4).contiguous()

  q_new = quat_mul(q_perturb, q_default_exp)  # (n_envs, n_entities, 4)

  model_field = getattr(env.sim.model, field)
  env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
  model_field[env_grid, entity_grid] = q_new
