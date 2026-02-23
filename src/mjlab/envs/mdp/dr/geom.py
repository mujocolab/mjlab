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
  _randomize_model_field,
  _randomize_quat_field,
)
from ._types import Distribution, Operation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# MuJoCo geom type integer constants.
_GEOM_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE.value
_GEOM_CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE.value
_GEOM_ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID.value
_GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER.value
_GEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX.value


def _recompute_geom_bounds(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Recompute ``geom_rbound`` and ``geom_aabb`` from current ``geom_size``.

  Only primitive types (sphere, capsule, ellipsoid, cylinder, box) are handled. Plane,
  hfield, mesh, and SDF geoms are left unchanged.
  """
  asset = env.scene[asset_cfg.name]
  entity_indices = _get_entity_indices(asset.indexing, asset_cfg, "geom", False)

  env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
  size = env.sim.model.geom_size[env_grid, entity_grid]  # (n_envs, n_geoms, 3)
  s0, s1, s2 = size[..., 0], size[..., 1], size[..., 2]

  # geom_type is (ngeom,) int, not per-world.
  geom_type_all = torch.as_tensor(env.sim.model.geom_type, device=env.device)
  gtype = geom_type_all[entity_indices]  # (G,) int

  # Compute rbound per type via torch.where cascades.
  rbound = torch.zeros_like(s0)

  is_sphere = gtype[None, :] == _GEOM_SPHERE
  is_capsule = gtype[None, :] == _GEOM_CAPSULE
  is_ellipsoid = gtype[None, :] == _GEOM_ELLIPSOID
  is_cylinder = gtype[None, :] == _GEOM_CYLINDER
  is_box = gtype[None, :] == _GEOM_BOX

  rbound = torch.where(is_sphere, s0, rbound)
  rbound = torch.where(is_capsule, s0 + s1, rbound)
  rbound = torch.where(is_ellipsoid, torch.maximum(s0, torch.maximum(s1, s2)), rbound)
  rbound = torch.where(is_cylinder, torch.sqrt(s0 * s0 + s1 * s1), rbound)
  rbound = torch.where(is_box, torch.sqrt(s0 * s0 + s1 * s1 + s2 * s2), rbound)

  # Compute aabb half-sizes per type.
  aabb_half_x = torch.zeros_like(s0)
  aabb_half_y = torch.zeros_like(s0)
  aabb_half_z = torch.zeros_like(s0)

  # Sphere: (s0, s0, s0)
  aabb_half_x = torch.where(is_sphere, s0, aabb_half_x)
  aabb_half_y = torch.where(is_sphere, s0, aabb_half_y)
  aabb_half_z = torch.where(is_sphere, s0, aabb_half_z)
  # Capsule: (s0, s0, s0+s1)
  aabb_half_x = torch.where(is_capsule, s0, aabb_half_x)
  aabb_half_y = torch.where(is_capsule, s0, aabb_half_y)
  aabb_half_z = torch.where(is_capsule, s0 + s1, aabb_half_z)
  # Cylinder: (s0, s0, s1)
  aabb_half_x = torch.where(is_cylinder, s0, aabb_half_x)
  aabb_half_y = torch.where(is_cylinder, s0, aabb_half_y)
  aabb_half_z = torch.where(is_cylinder, s1, aabb_half_z)
  # Ellipsoid: (s0, s1, s2)
  aabb_half_x = torch.where(is_ellipsoid, s0, aabb_half_x)
  aabb_half_y = torch.where(is_ellipsoid, s1, aabb_half_y)
  aabb_half_z = torch.where(is_ellipsoid, s2, aabb_half_z)
  # Box: (s0, s1, s2)
  aabb_half_x = torch.where(is_box, s0, aabb_half_x)
  aabb_half_y = torch.where(is_box, s1, aabb_half_y)
  aabb_half_z = torch.where(is_box, s2, aabb_half_z)

  is_supported = is_sphere | is_capsule | is_ellipsoid | is_cylinder | is_box

  if not is_supported.all():
    unsupported = gtype[~is_supported[0]]
    names = [mujoco.mjtGeom(t.item()).name for t in unsupported]
    raise ValueError(
      f"dr.geom_size only supports primitive geom types (sphere, capsule, "
      f"ellipsoid, cylinder, box). Got unsupported types: {names}"
    )

  env.sim.model.geom_rbound[env_grid, entity_grid] = rbound

  # aabb shape is (*, ngeom, 2, 3) — [0] is center, [1] is half-size.
  # Center stays (0,0,0) for all primitives; only update half-size.
  aabb_half = torch.stack([aabb_half_x, aabb_half_y, aabb_half_z], dim=-1)
  env.sim.model.geom_aabb[env_grid, entity_grid, 1] = aabb_half


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
  """Randomize geom RGBA colors."""
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


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
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
  """Randomize geom size and recompute broadphase bounds.

  After writing new values to ``geom_size``, this function automatically recomputes
  ``geom_rbound`` (bounding sphere) and ``geom_aabb`` (local bounding box) so that the
  broadphase remains consistent. Only primitive geom types (sphere, capsule, ellipsoid,
  cylinder, box) are supported.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  _randomize_model_field(
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
  _recompute_geom_bounds(env, env_ids, asset_cfg)
