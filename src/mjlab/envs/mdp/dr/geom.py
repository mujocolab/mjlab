"""Domain randomization functions for geom fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import (
  _DEFAULT_ASSET_CFG,
  Ranges,
  _get_entity_indices,
  _randomize_categorical_field,
  _randomize_model_field,
  _randomize_quat_field,
)
from ._types import Distribution, Operation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Geom types whose bounds set_const derives from geom_size.
_PRIMITIVE_GEOMS = frozenset(
  int(t)
  for t in (
    mujoco.mjtGeom.mjGEOM_SPHERE,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_BOX,
  )
)


# Per-field functions.


@requires_model_fields("geom_friction")
def geom_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom friction.

  Default axis is 0 (tangential friction only). Axes 1 (torsional) and 2 (rolling) only
  affect contacts with ``condim >= 4``; pass ``axes=[0, 1, 2]`` explicitly if your
  model uses high-dimensional contacts.
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
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom positions."""
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
  roll_range: tuple[float, float] = (0.0, 0.0),
  pitch_range: tuple[float, float] = (0.0, 0.0),
  yaw_range: tuple[float, float] = (0.0, 0.0),
  distribution: Distribution | str = "uniform",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize geom orientation by composing an RPY perturbation.

  Ranges are in radians. The sampled perturbation is composed with the default
  quaternion (not the current one), so repeated calls do not accumulate. The result is
  always a valid unit quaternion.
  """
  _randomize_quat_field(
    env,
    env_ids,
    "geom_quat",
    entity_type="geom",
    roll_range=roll_range,
    pitch_range=pitch_range,
    yaw_range=yaw_range,
    distribution=distribution,
    asset_cfg=asset_cfg,
  )


@requires_model_fields("geom_rgba")
def geom_rgba(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom RGBA colors.

  Only affects rendering when the geom has no material assigned. If a material is
  assigned, its color (or texture) takes precedence over ``geom_rgba``.
  Use :func:`~mjlab.envs.mdp.dr.mat_rgba` instead, or clear the material assignment
  first.

  Args:
    env: The environment instance.
    env_ids: Environment indices to randomize. ``None`` means all.
    ranges: Value range(s) for sampling.
    asset_cfg: Entity and geom selection.
    distribution: Sampling distribution.
    operation: How to combine sampled values with the base.
    axes: Which RGBA channels to randomize. Defaults to ``[0, 1, 2, 3]``.
    shared_random: If ``True``, all selected geoms receive the same sampled value per
      environment.
  """
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


@requires_model_fields("geom_size")
def geom_size(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "scale",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom size. The EventManager then recomputes the broadphase bounds
  (geom_rbound and geom_aabb).

  Only primitive geom types (sphere, capsule, ellipsoid, cylinder, box) are supported.
  """
  geom_ids = _randomize_model_field(
    env,
    env_ids,
    "geom_size",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )
  geom_types = env.sim.mj_model.geom_type[geom_ids.cpu().numpy()]
  unsupported = [
    mujoco.mjtGeom(t).name for t in geom_types if t not in _PRIMITIVE_GEOMS
  ]
  if unsupported:
    raise ValueError(
      "dr.geom_size only supports primitive geom types (sphere, capsule, "
      f"ellipsoid, cylinder, box). Got unsupported types: {unsupported}"
    )


@requires_model_fields("geom_matid")
def geom_matid(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  shared_random: bool = False,
) -> None:
  """Randomize which baked material each selected geom uses.

  Each geom is assigned a material drawn uniformly from ``asset_cfg.material_names``.
  Assigning a material to a geom that had none (``matid < 0``) switches its color
  source from ``geom_rgba`` to the material's ``mat_rgba``/texture.

  Args:
    env: The environment instance.
    env_ids: Environment indices to randomize. ``None`` means all.
    asset_cfg: Entity, geom, and material selection.
    shared_random: If ``True``, all selected geoms receive the same sampled value per
      environment.
  """
  asset = env.scene[asset_cfg.name]
  mat_ids = _get_entity_indices(asset.indexing, asset_cfg, "material", False)
  if mat_ids.numel() == 0:
    raise ValueError(
      f"No materials selected for entity '{asset_cfg.name}'. Set "
      "SceneEntityCfg.material_names to a non-empty selection."
    )

  _randomize_categorical_field(
    env,
    env_ids,
    "geom_matid",
    entity_type="geom",
    pool=mat_ids,
    asset_cfg=asset_cfg,
    shared_random=shared_random,
  )
