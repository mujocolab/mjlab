"""Per-world mesh variant support.

Builds on mujoco_warp's batched ``geom_dataid`` to assign different meshes
to different simulation worlds. Adapted from the upstream reference
implementation in ``mujoco_warp/_src/io_test.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from mjlab.entity.entity import VariantMetadata

# Fields that depend on mesh geometry and must be compiled per-variant.
VARIANT_DEPENDENT_FIELDS = (
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "geom_pos",
  "geom_quat",
  "body_mass",
  "body_subtreemass",
  "body_inertia",
  "body_invweight0",
  "body_ipos",
  "body_iquat",
)


@dataclass
class PerWorldMeshResult:
  """Output of :func:`per_world_mesh`."""

  wp_model: mjwarp.Model
  mj_model: mujoco.MjModel
  # Maps entity prefix -> array of variant indices per world.
  world_to_variant: dict[str, np.ndarray]


def _find_entity_mesh_geom_ids(
  model: mujoco.MjModel,
  entity_prefix: str,
) -> list[int]:
  """Find all mesh geom IDs belonging to an entity, including padding."""
  named_ids: list[int] = []
  for gid in range(model.ngeom):
    gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    if (
      gname
      and gname.startswith(entity_prefix)
      and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
    ):
      named_ids.append(gid)
  if not named_ids:
    return []
  # Include unnamed padding geoms on the same body.
  body_id = model.geom_bodyid[named_ids[0]]
  all_ids = set(named_ids)
  for gid in range(model.ngeom):
    if (
      model.geom_bodyid[gid] == body_id
      and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
    ):
      all_ids.add(gid)
  return sorted(all_ids)


def allocate_worlds(
  weights: tuple[float, ...],
  nworld: int,
) -> list[int]:
  """Assign worlds proportionally by weight (largest-remainder method).

  Returns a list of length *nworld* containing variant indices.
  """
  total = sum(weights)
  if total <= 0:
    total = float(len(weights))
    weights = tuple(1.0 for _ in weights)
  quotas = [(w / total) * nworld for w in weights]
  floors = [int(q) for q in quotas]
  remainders = sorted(
    ((quotas[i] - floors[i], i) for i in range(len(weights))),
    key=lambda x: -x[0],
  )
  allocated = sum(floors)
  for j in range(nworld - allocated):
    floors[remainders[j][1]] += 1
  assignment: list[int] = []
  for idx, count in enumerate(floors):
    assignment.extend([idx] * count)
  return assignment


def per_world_mesh(
  spec: mujoco.MjSpec,
  nworld: int,
  variant_info: list[tuple[str, VariantMetadata]],
  configure_model: Callable[[mujoco.MjModel], None] | None = None,
) -> PerWorldMeshResult:
  """Build a warp Model with per-world mesh assignments.

  Args:
    spec: Scene spec (already merged with padded variant geoms).
    nworld: Number of simulation worlds.
    variant_info: List of ``(entity_prefix, metadata)`` pairs for
      entities that have mesh variants.
    configure_model: Optional callback to configure the compiled
      MjModel before ``put_model`` (e.g., setting solver options).

  Returns:
    A :class:`PerWorldMeshResult` containing the warp model, host
    model, and per-entity world-to-variant mappings.
  """
  spec = spec.copy()
  model = spec.compile()
  if configure_model is not None:
    configure_model(model)

  # Start from base dataid tiled for all worlds.
  base_dataid = model.geom_dataid.copy()
  dataid_table = np.tile(base_dataid, (nworld, 1))

  world_to_variant: dict[str, np.ndarray] = {}

  for entity_prefix, metadata in variant_info:
    # Allocate worlds by weight.
    assignment = allocate_worlds(metadata.variant_weights, nworld)
    w2v = np.array(assignment, dtype=np.int32)
    world_to_variant[entity_prefix] = w2v

    mesh_geom_ids = _find_entity_mesh_geom_ids(model, entity_prefix)

    # Set per-world dataid for each mesh geom slot.
    for w in range(nworld):
      variant_idx = assignment[w]
      mesh_names = metadata.variant_mesh_names[variant_idx]
      for slot, geom_id in enumerate(mesh_geom_ids):
        if slot < len(mesh_names) and mesh_names[slot] is not None:
          # Resolve prefixed mesh name to mesh ID.
          mesh_name = mesh_names[slot]
          assert mesh_name is not None
          # The mesh name in the merged spec is already prefixed
          # with the variant name (e.g., "mug/visual_mesh"). But
          # when attached to the scene, it also gets the entity
          # prefix (e.g., "object/mug/visual_mesh"). Resolve using
          # the entity prefix.
          full_mesh_name = f"{entity_prefix}{mesh_name}"
          mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, full_mesh_name)
          if mesh_id < 0:
            raise ValueError(f"Mesh '{full_mesh_name}' not found in compiled model.")
          dataid_table[w, geom_id] = mesh_id
        else:
          # Padding slot: disable this geom for this world.
          dataid_table[w, geom_id] = -1

  # Build warp model.
  m = mjwarp.put_model(model)
  m.geom_dataid = wp.array(dataid_table, dtype=int)

  # Populate dependent per-world fields.
  _populate_dependent_fields(m, spec, model, dataid_table, nworld, variant_info)

  return PerWorldMeshResult(
    wp_model=m,
    mj_model=model,
    world_to_variant=world_to_variant,
  )


def _populate_dependent_fields(
  m: mjwarp.Model,
  spec: mujoco.MjSpec,
  padded_model: mujoco.MjModel,
  dataid_table: np.ndarray,
  nworld: int,
  variant_info: list[tuple[str, VariantMetadata]],
) -> None:
  """Compile each unique variant and write per-world dependent fields.

  Mutates *spec* temporarily (saves and restores geom state).
  """
  # Find unique dataid rows.
  unique_rows: dict[tuple[int, ...], int] = {}
  for w in range(nworld):
    key = tuple(dataid_table[w])
    if key not in unique_rows:
      unique_rows[key] = w

  if len(unique_rows) <= 1:
    return  # All worlds identical, nothing to do.

  # Save spec geom state.
  spec_geoms = list(spec.geoms)
  saved: dict[int, tuple[str, int, int, float]] = {}
  for idx, g in enumerate(spec_geoms):
    saved[idx] = (g.meshname, g.contype, g.conaffinity, g.mass)

  # Map geom IDs in padded_model to spec geom indices.
  geom_id_to_spec_idx: dict[int, int] = {}
  for idx, g in enumerate(spec_geoms):
    if g.name:
      gid = mujoco.mj_name2id(padded_model, mujoco.mjtObj.mjOBJ_GEOM, g.name)
      if gid >= 0:
        geom_id_to_spec_idx[gid] = idx

  # Collect all variant geom IDs.
  all_variant_geom_ids: set[int] = set()
  for entity_prefix, _ in variant_info:
    all_variant_geom_ids.update(_find_entity_mesh_geom_ids(padded_model, entity_prefix))

  # Compile each unique variant.
  compiled_variants: dict[tuple[int, ...], mujoco.MjModel] = {}
  for key, first_world in unique_rows.items():
    for gid in all_variant_geom_ids:
      if gid not in geom_id_to_spec_idx:
        continue
      spec_idx = geom_id_to_spec_idx[gid]
      mesh_id = dataid_table[first_world, gid]
      if mesh_id >= 0:
        mesh_name = mujoco.mj_id2name(padded_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        spec_geoms[spec_idx].meshname = mesh_name
        spec_geoms[spec_idx].contype = 1
        spec_geoms[spec_idx].conaffinity = 1
      else:
        spec_geoms[spec_idx].contype = 0
        spec_geoms[spec_idx].conaffinity = 0
        spec_geoms[spec_idx].mass = 0.0
    compiled_variants[key] = spec.compile()

  # Restore spec state.
  for idx, g in enumerate(spec_geoms):
    if idx in saved:
      meshname, contype, conaffinity, mass = saved[idx]
      g.meshname = meshname
      g.contype = contype
      g.conaffinity = conaffinity
      g.mass = mass

  # Build per-world numpy arrays.
  ngeom = padded_model.ngeom
  nbody = padded_model.nbody

  geom_size = np.zeros((nworld, ngeom, 3), dtype=np.float32)
  geom_rbound = np.zeros((nworld, ngeom), dtype=np.float32)
  geom_aabb = np.zeros((nworld, ngeom, 2, 3), dtype=np.float32)
  geom_pos = np.zeros((nworld, ngeom, 3), dtype=np.float32)
  geom_quat = np.zeros((nworld, ngeom, 4), dtype=np.float32)
  body_mass = np.zeros((nworld, nbody), dtype=np.float32)
  body_subtreemass = np.zeros((nworld, nbody), dtype=np.float32)
  body_inertia = np.zeros((nworld, nbody, 3), dtype=np.float32)
  body_invweight0 = np.zeros((nworld, nbody, 2), dtype=np.float32)
  body_ipos = np.zeros((nworld, nbody, 3), dtype=np.float32)
  body_iquat = np.zeros((nworld, nbody, 4), dtype=np.float32)

  for w in range(nworld):
    key = tuple(dataid_table[w])
    ref = compiled_variants[key]
    geom_size[w] = ref.geom_size
    geom_rbound[w] = ref.geom_rbound
    geom_aabb[w] = ref.geom_aabb.reshape(ngeom, 2, 3)
    geom_pos[w] = ref.geom_pos
    geom_quat[w] = ref.geom_quat
    body_mass[w] = ref.body_mass
    body_subtreemass[w] = ref.body_subtreemass
    body_inertia[w] = ref.body_inertia
    body_invweight0[w] = ref.body_invweight0
    body_ipos[w] = ref.body_ipos
    body_iquat[w] = ref.body_iquat

  m.geom_size = wp.array(geom_size, dtype=wp.vec3)
  m.geom_rbound = wp.array(geom_rbound, dtype=float)
  m.geom_aabb = wp.array(geom_aabb, dtype=wp.vec3)
  m.geom_pos = wp.array(geom_pos, dtype=wp.vec3)
  m.geom_quat = wp.array(geom_quat, dtype=wp.quat)
  m.body_mass = wp.array(body_mass, dtype=float)
  m.body_subtreemass = wp.array(body_subtreemass, dtype=float)
  m.body_inertia = wp.array(body_inertia, dtype=wp.vec3)
  m.body_invweight0 = wp.array(body_invweight0, dtype=wp.vec2)
  m.body_ipos = wp.array(body_ipos, dtype=wp.vec3)
  m.body_iquat = wp.array(body_iquat, dtype=wp.quat)
