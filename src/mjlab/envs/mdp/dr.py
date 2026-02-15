"""Domain randomization functions for model fields.

Import this module and use functions as ``dr.geom_friction(...)``,
``dr.body_mass(...)``, etc.

Example::

    from mjlab.envs.mdp import dr

    foot_friction = EventTermCfg(
        mode="reset",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_foot.*",)),
            "ranges": (0.3, 1.2),
            "operation": "abs",
        },
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity, EntityIndexing
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  sample_gaussian,
  sample_log_uniform,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _randomize_model_field(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  field: str,
  *,
  entity_type: Literal["dof", "joint", "body", "geom", "site", "actuator", "tendon"],
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  distribution: str = "uniform",
  operation: str = "abs",
  asset_cfg: SceneEntityCfg,
  axes: list[int] | None = None,
  shared_random: bool = False,
  default_axes: list[int] | None = None,
  valid_axes: list[int] | None = None,
  use_address: bool = False,
) -> None:
  """Core model-field randomization engine (private).

  Handles scalar and multi-axis fields, scale/add/abs operations with
  default-based accumulation prevention, shared_random broadcasting,
  and per-component string-keyed ranges.
  """
  if isinstance(ranges, dict) and not ranges:
    return

  # Per-component ranges: string keys -> resolve each pattern separately.
  if isinstance(ranges, dict) and isinstance(next(iter(ranges.keys())), str):
    _names_attr = _ENTITY_NAMES_ATTR[entity_type]
    for pattern, pattern_range in ranges.items():
      sub_cfg = SceneEntityCfg(asset_cfg.name)
      setattr(sub_cfg, _names_attr, (pattern,))
      sub_cfg.resolve(env.scene)
      _randomize_model_field(
        env,
        env_ids,
        field,
        entity_type=entity_type,
        ranges=pattern_range,
        distribution=distribution,
        operation=operation,
        asset_cfg=sub_cfg,
        axes=axes,
        shared_random=shared_random,
        default_axes=default_axes,
        valid_axes=valid_axes,
        use_address=use_address,
      )
    return

  # After the string-keyed branch, ranges is tuple or int-keyed dict.
  assert not isinstance(ranges, dict) or isinstance(next(iter(ranges.keys())), int)
  int_ranges: tuple[float, float] | dict[int, tuple[float, float]] = ranges  # type: ignore[assignment]

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
    model_field, axes, int_ranges, default_axes, valid_axes
  )

  axis_ranges = _prepare_axis_ranges(int_ranges, target_axes, field)

  env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
  indexed_data = model_field[env_grid, entity_grid]

  # For scale/add operations, use stored default values to prevent
  # accumulation.
  if operation in ("scale", "add"):
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

  _apply_operation(
    model_field, env_grid, entity_grid, base_values, random_values, operation
  )


# Maps entity_type to the SceneEntityCfg names attribute for per-component
# range resolution.
_ENTITY_NAMES_ATTR: dict[str, str] = {
  "dof": "joint_names",
  "joint": "joint_names",
  "body": "body_names",
  "geom": "geom_names",
  "site": "site_names",
  "actuator": "actuator_names",
  "tendon": "tendon_names",
}


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
    case _:
      raise ValueError(f"Unknown entity type: {entity_type}")


def _determine_target_axes(
  model_field: torch.Tensor,
  axes: list[int] | None,
  ranges: tuple[float, float] | dict[int, tuple[float, float]],
  default_axes: list[int] | None,
  valid_axes: list[int] | None,
) -> list[int]:
  """Determine which axes to randomize."""
  field_ndim = len(model_field.shape) - 1  # Subtract env dimension

  if axes is not None:
    target_axes = axes
  elif isinstance(ranges, dict):
    target_axes = list(ranges.keys())
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
  axis_ranges: dict[int, tuple[float, float]],
  indexed_data: torch.Tensor,
  target_axes: list[int],
  device: str,
  operation: str,
) -> torch.Tensor:
  """Generate random values for the specified axes.

  For scale/add operations, non-randomized axes use identity values (1.0
  for scale, 0.0 for add) to prevent modification. For abs operations,
  non-randomized axes preserve their current values.
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


# ---------------------------------------------------------------------------
# Geom
# ---------------------------------------------------------------------------


@requires_model_fields("geom_friction")
def geom_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom friction coefficients.

  Geoms have three friction coefficients ``[tangential, torsional,
  rolling]``. For ``condim=3`` (standard frictional contact), only axis 0
  (tangential) affects contact behavior. By default only axis 0 is
  randomized.
  """
  _randomize_model_field(
    env,
    env_ids,
    "geom_friction",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0],
    valid_axes=[0, 1, 2],
  )


@requires_model_fields("geom_pos")
def geom_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom positions (geom_pos)."""
  _randomize_model_field(
    env,
    env_ids,
    "geom_pos",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("geom_quat")
def geom_quat(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom orientation quaternions (geom_quat)."""
  _randomize_model_field(
    env,
    env_ids,
    "geom_quat",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2, 3],
  )


@requires_model_fields("geom_rgba")
def geom_rgba(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom RGBA colors (geom_rgba)."""
  _randomize_model_field(
    env,
    env_ids,
    "geom_rgba",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2, 3],
  )


# ---------------------------------------------------------------------------
# Site
# ---------------------------------------------------------------------------


@requires_model_fields("site_pos")
def site_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize site positions (site_pos)."""
  _randomize_model_field(
    env,
    env_ids,
    "site_pos",
    entity_type="site",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("site_quat")
def site_quat(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize site orientation quaternions (site_quat)."""
  _randomize_model_field(
    env,
    env_ids,
    "site_quat",
    entity_type="site",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2, 3],
  )


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


@requires_model_fields("body_mass", recompute="set_const")
def body_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "scale",
  shared_random: bool = False,
) -> None:
  """Randomize body mass. Derived quantities are recomputed via set_const."""
  _randomize_model_field(
    env,
    env_ids,
    "body_mass",
    entity_type="body",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("body_inertia", recompute="set_const_0")
def body_inertia(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "scale",
  shared_random: bool = False,
) -> None:
  """Randomize body inertia. Recomputed via set_const_0."""
  _randomize_model_field(
    env,
    env_ids,
    "body_inertia",
    entity_type="body",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("body_iquat", recompute="set_const_0")
def body_inertia_quat(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize body inertia orientation quaternion (body_iquat)."""
  _randomize_model_field(
    env,
    env_ids,
    "body_iquat",
    entity_type="body",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2, 3],
  )


@requires_model_fields("body_ipos", recompute="set_const")
def body_com_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  shared_random: bool = False,
) -> None:
  """Randomize body center-of-mass offset (body_ipos)."""
  _randomize_model_field(
    env,
    env_ids,
    "body_ipos",
    entity_type="body",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    default_axes=[0, 1, 2],
    shared_random=shared_random,
  )


@requires_model_fields("body_pos", recompute="set_const_0")
def body_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  shared_random: bool = False,
) -> None:
  """Randomize body position. Recomputed via set_const_0."""
  _randomize_model_field(
    env,
    env_ids,
    "body_pos",
    entity_type="body",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    default_axes=[0, 1, 2],
    shared_random=shared_random,
  )


@requires_model_fields("body_quat", recompute="set_const_0")
def body_quat(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: (
    tuple[float, float]
    | dict[int, tuple[float, float]]
    | dict[str, tuple[float, float]]
  ),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  shared_random: bool = False,
) -> None:
  """Randomize body orientation quaternion. Recomputed via set_const_0."""
  _randomize_model_field(
    env,
    env_ids,
    "body_quat",
    entity_type="body",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    default_axes=[0, 1, 2, 3],
    shared_random=shared_random,
  )


# ---------------------------------------------------------------------------
# Joint / DOF
# ---------------------------------------------------------------------------


@requires_model_fields("dof_damping")
def joint_damping(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  shared_random: bool = False,
) -> None:
  """Randomize joint damping (dof_damping)."""
  _randomize_model_field(
    env,
    env_ids,
    "dof_damping",
    entity_type="dof",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("dof_armature", recompute="set_const_0")
def joint_armature(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  shared_random: bool = False,
) -> None:
  """Randomize joint armature. Recomputed via set_const_0."""
  _randomize_model_field(
    env,
    env_ids,
    "dof_armature",
    entity_type="dof",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("dof_frictionloss")
def joint_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  shared_random: bool = False,
) -> None:
  """Randomize joint friction loss (dof_frictionloss)."""
  _randomize_model_field(
    env,
    env_ids,
    "dof_frictionloss",
    entity_type="dof",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("jnt_stiffness")
def joint_stiffness(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  shared_random: bool = False,
) -> None:
  """Randomize joint stiffness (jnt_stiffness)."""
  _randomize_model_field(
    env,
    env_ids,
    "jnt_stiffness",
    entity_type="joint",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("jnt_range")
def joint_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  shared_random: bool = False,
) -> None:
  """Randomize joint position limits (jnt_range)."""
  _randomize_model_field(
    env,
    env_ids,
    "jnt_range",
    entity_type="joint",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("qpos0", recompute="set_const_0")
def joint_default_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "add",
  shared_random: bool = False,
) -> None:
  """Randomize default joint positions (qpos0). Recomputed via set_const_0."""
  _randomize_model_field(
    env,
    env_ids,
    "qpos0",
    entity_type="joint",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    use_address=True,
    shared_random=shared_random,
  )


# ---------------------------------------------------------------------------
# Tendon
# ---------------------------------------------------------------------------


@requires_model_fields("tendon_damping")
def tendon_damping(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  shared_random: bool = False,
) -> None:
  """Randomize tendon damping."""
  _randomize_model_field(
    env,
    env_ids,
    "tendon_damping",
    entity_type="tendon",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("tendon_stiffness")
def tendon_stiffness(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  shared_random: bool = False,
) -> None:
  """Randomize tendon stiffness."""
  _randomize_model_field(
    env,
    env_ids,
    "tendon_stiffness",
    entity_type="tendon",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("tendon_frictionloss")
def tendon_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  shared_random: bool = False,
) -> None:
  """Randomize tendon friction loss (tendon_frictionloss)."""
  _randomize_model_field(
    env,
    env_ids,
    "tendon_frictionloss",
    entity_type="tendon",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


@requires_model_fields("tendon_lengthspring")
def tendon_length_spring(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  shared_random: bool = False,
) -> None:
  """Randomize tendon spring rest length (tendon_lengthspring)."""
  _randomize_model_field(
    env,
    env_ids,
    "tendon_lengthspring",
    entity_type="tendon",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )


# ---------------------------------------------------------------------------
# Actuator
# ---------------------------------------------------------------------------


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize PD stiffness and damping gains.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    kp_range: (min, max) for proportional gain randomization.
    kd_range: (min, max) for derivative gain randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default gains by sampled values, "abs"
      sets absolute values.
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
    ctrl_ids = actuator.global_ctrl_ids

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
        f"pd_gains only supports BuiltinPositionActuator,"
        f" XmlPositionActuator, and IdealPdActuator (optionally"
        f" wrapped with DelayedActuator),"
        f" got {type(actuator).__name__}"
      )


@requires_model_fields("actuator_forcerange")
def effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize actuator effort limits.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    effort_limit_range: (min, max) for effort limit randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies existing limits, "abs" sets absolute.
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
    ctrl_ids = actuator.global_ctrl_ids
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
        default_forcerange = env.sim.get_default_field("actuator_forcerange")
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
          default_forcerange[ctrl_ids, 0] * effort_samples
        )
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          default_forcerange[ctrl_ids, 1] * effort_samples
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
        assert actuator.default_force_limit is not None
        actuator.set_effort_limit(
          env_ids,
          effort_limit=actuator.default_force_limit[env_ids] * effort_samples,
        )
      elif operation == "abs":
        actuator.set_effort_limit(env_ids, effort_limit=effort_samples)

    else:
      raise TypeError(
        f"effort_limits only supports BuiltinPositionActuator,"
        f" XmlPositionActuator, and IdealPdActuator,"
        f" got {type(actuator).__name__}"
      )


# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------


def encoder_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  bias_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize encoder bias to simulate joint encoder calibration errors.

  See docs/source/randomization.rst for details on how encoder bias works.
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


def actuator_delays(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  lag_range: tuple[int, int],
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
