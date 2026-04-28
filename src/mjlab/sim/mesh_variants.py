"""Per-world mesh variant support.

Sibling of :mod:`mjlab.sim.randomization`: that module expands singleton
model fields into per-world arrays for DR; this one writes per-world
arrays whose rows differ by mesh variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from mjlab.entity.entity import BodyInertialMetadata, VariantMetadata

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
class MeshVariantResult:
  """Output of :func:`build_mesh_variant_model`."""

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

  Returns a list of length *nworld* containing variant indices. Weights
  must be non-negative with at least one positive entry.
  """
  if any(w < 0 for w in weights):
    raise ValueError(f"weights must be non-negative, got {weights}.")
  total = sum(weights)
  if total <= 0:
    raise ValueError(f"weights must have a positive sum, got {weights}.")
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


def build_mesh_variant_model(
  spec: mujoco.MjSpec,
  nworld: int,
  variant_info: list[tuple[str, VariantMetadata]],
  configure_model: Callable[[mujoco.MjModel], None] | None = None,
) -> MeshVariantResult:
  """Build a warp Model with per-world mesh assignments.

  Args:
    spec: Scene spec (already merged with padded variant geoms).
    nworld: Number of simulation worlds.
    variant_info: List of ``(entity_prefix, metadata)`` pairs for
      entities that have mesh variants.
    configure_model: Optional callback to configure the compiled
      MjModel before ``put_model`` (e.g., setting solver options).

  Returns:
    A :class:`MeshVariantResult` containing the warp model, host
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
    nslots = len(mesh_geom_ids)

    # Resolve every (variant, slot) -> mesh_id once. Mesh names in the merged
    # spec are variant-prefixed ("mug/visual_mesh"); after attaching to the
    # scene they also carry the entity prefix ("object/mug/visual_mesh").
    # Padding slots are -1.
    nvariants = len(metadata.variant_mesh_names)
    variant_slot_ids = np.full((nvariants, nslots), -1, dtype=np.int64)
    for v_idx, mesh_names in enumerate(metadata.variant_mesh_names):
      for slot in range(min(nslots, len(mesh_names))):
        name = mesh_names[slot]
        if name is None:
          continue
        full = f"{entity_prefix}{name}"
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, full)
        if mid < 0:
          variant_label = metadata.variant_names[v_idx]
          raise ValueError(
            f"Mesh '{full}' (variant '{variant_label}', slot {slot}) "
            f"not found in compiled model."
          )
        variant_slot_ids[v_idx, slot] = mid

    # Vectorized scatter: row-select by variant assignment, write into the
    # mesh-geom columns of the per-world dataid table.
    dataid_table[:, mesh_geom_ids] = variant_slot_ids[w2v]

  # Build warp model.
  m = mjwarp.put_model(model)
  m.geom_dataid = wp.array(dataid_table, dtype=int)

  # Populate dependent per-world fields.
  _populate_dependent_fields(
    m, spec, model, dataid_table, nworld, variant_info, world_to_variant
  )

  return MeshVariantResult(
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
  world_to_variant: dict[str, np.ndarray],
) -> None:
  """Compile each unique variant and write per-world dependent fields.

  Each unique variant is compiled from a fresh ``spec.copy()``; the input
  ``spec`` is not mutated.
  """
  # Find unique dataid rows.
  unique_rows: dict[tuple[int, ...], int] = {}
  for w in range(nworld):
    key = tuple(dataid_table[w])
    if key not in unique_rows:
      unique_rows[key] = w

  # Map padded_model geom IDs to geom names (stable across spec copies).
  geom_id_to_name: dict[int, str] = {}
  for g in spec.geoms:
    if not g.name:
      continue
    gid = mujoco.mj_name2id(padded_model, mujoco.mjtObj.mjOBJ_GEOM, g.name)
    if gid >= 0:
      geom_id_to_name[gid] = g.name

  # Collect all variant geom IDs in padded_model.
  all_variant_geom_ids: set[int] = set()
  for entity_prefix, _ in variant_info:
    all_variant_geom_ids.update(_find_entity_mesh_geom_ids(padded_model, entity_prefix))

  # Bodies any variant marks as explicit-inertial: must be reset on the
  # fresh spec copy before applying this variant's inertials. Variants
  # without an explicit inertial fall back to MuJoCo's mesh-derived path
  # during compile, so we clear the diagonal inertial fields. Do NOT
  # assign ``body.fullinertia``: any assignment (even zeros) flags the
  # field as user-specified and ``spec.compile()`` then rejects it as
  # conflicting with ``body.inertia``.
  variant_inertial_body_names: set[str] = set()
  for entity_prefix, metadata in variant_info:
    for variant_inertials in metadata.variant_body_inertials:
      for inertial in variant_inertials:
        variant_inertial_body_names.add(f"{entity_prefix}{inertial.body_name}")

  # Compile each unique variant from a fresh spec copy.
  compiled_variants: dict[tuple[int, ...], mujoco.MjModel] = {}
  for key, first_world in unique_rows.items():
    variant_spec = spec.copy()
    geoms_by_name = {g.name: g for g in variant_spec.geoms if g.name}
    bodies_by_name = {b.name: b for b in variant_spec.bodies if b.name}

    # Apply this variant's mesh selection per geom slot.
    for gid in all_variant_geom_ids:
      name = geom_id_to_name.get(gid)
      if name is None:
        continue
      geom = geoms_by_name[name]
      mesh_id = int(dataid_table[first_world, gid])
      if mesh_id >= 0:
        mesh_name = mujoco.mj_id2name(padded_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        geom.meshname = mesh_name
        geom.contype = 1
        geom.conaffinity = 1
      else:
        geom.contype = 0
        geom.conaffinity = 0
        geom.mass = 0.0

    for body_name in variant_inertial_body_names:
      body = bodies_by_name.get(body_name)
      if body is None:
        continue
      body.explicitinertial = 0
      body.mass = 0.0
      body.inertia = np.zeros(3, dtype=np.float64)
      body.ipos = np.zeros(3, dtype=np.float64)
      body.iquat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for entity_prefix, metadata in variant_info:
      variant_idx = int(world_to_variant[entity_prefix][first_world])
      if variant_idx >= len(metadata.variant_body_inertials):
        continue
      for inertial in metadata.variant_body_inertials[variant_idx]:
        _apply_body_inertial(
          bodies_by_name,
          f"{entity_prefix}{inertial.body_name}",
          inertial,
        )

    compiled_variants[key] = variant_spec.compile()

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


def _apply_body_inertial(
  bodies_by_name: dict[str, mujoco.MjsBody],
  body_name: str,
  inertial: BodyInertialMetadata,
) -> None:
  body = bodies_by_name.get(body_name)
  if body is None:
    raise ValueError(f"Body '{body_name}' not found in compiled variant spec.")
  body.explicitinertial = 1
  body.mass = inertial.mass
  body.ipos = np.asarray(inertial.ipos, dtype=np.float64)
  body.inertia = np.asarray(inertial.inertia, dtype=np.float64)
  body.iquat = np.asarray(inertial.iquat, dtype=np.float64)
