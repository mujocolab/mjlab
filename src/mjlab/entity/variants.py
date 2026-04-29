"""Per-world mesh variant support.

A single batched simulation can run with different mesh assets in different
parallel worlds. World 0 may simulate a cube, world 1 a sphere, world 2 a
bowl, all sharing the same compiled scene and the same kinematic structure.
The result is a heterogeneous batch in which the mesh and its derived
constants vary across worlds while everything else (body tree, joint
structure, contact and solver setup) is fixed.

The feature spans two phases, both of which live in this file:

1. **Authoring** (entity-build time, MjSpec only). The user supplies a
   ``VariantEntityCfg`` whose ``variants`` map names to spec callables
   instances, each with its own ``spec_fn``. ``build_merged_variant_spec``
   validates that all source specs share kinematic topology, computes
   ``(body_path, role, ordinal)`` slots so each variable mesh geom has a
   structural identity, and merges the source specs into one padded
   template spec. It returns ``(template_spec, VariantMetadata)``.
2. **Realization** (sim-init time, mujoco_warp). ``build_variant_model``
   takes the scene-attached padded spec plus the per-entity
   ``VariantMetadata`` and produces a heterogeneous Warp model: a per-world
   ``geom_dataid`` table plus per-world arrays for the geometry-dependent
   fields listed in ``VARIANT_DEPENDENT_FIELDS``. Each unique variant is
   compiled once on the host with its source-spec geom semantics restored,
   and the resulting reference fields are scattered into per-world arrays
   keyed by world-to-variant assignment.

Slot identity is the throughline. ``SlotKey(body_path, role, ordinal)`` is
fixed across variants because mujoco_warp's ``geom_contype``/
``geom_conaffinity`` are 1D shared, so each slot's role (visual vs
collision) cannot change per world. ``VariantGeomSpec`` records each
variant's source-geom attributes per slot so the per-variant reference
compile restores them verbatim instead of inheriting whatever the template
variant set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence, cast

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from mjlab.entity.entity import EntityCfg
from mjlab.utils.mujoco import dof_width, qpos_width
from mjlab.utils.spec import copy_mesh_data

# Reserved name prefixes for synthesized template entities. Source variant
# specs must not create geoms, bodies, meshes, or other named items under
# these prefixes; validation rejects them so the merge pass can introduce
# padding geoms without colliding with user names.
RESERVED_NAME_PREFIXES: tuple[str, ...] = ("mjlab/pad/",)

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

GeomRole = Literal["visual", "collision"]
InertialMode = Literal["mesh-derived", "diagonal", "fullinertia"]


##
# Input: what the user writes.
##


def _variant_spec_fn_unset() -> mujoco.MjSpec:
  """Sentinel default for ``VariantEntityCfg.spec_fn``.

  ``VariantEntityCfg`` builds its spec from ``variants`` via
  ``build_merged_variant_spec``; the inherited ``spec_fn`` field is unused.
  Identity comparison against this sentinel detects accidental user overrides.
  """
  raise AssertionError(
    "VariantEntityCfg.spec_fn should never be called; the merged spec is "
    "built from `variants`."
  )


@dataclass
class VariantEntityCfg(EntityCfg):
  """Entity config for per-world mesh variants.

  Provide a dict of named variants (each value is a callable returning
  an ``MjSpec``) and optionally an ``assignment`` describing how worlds
  map to variants. The merged spec (with all variant meshes and padded
  geoms) is built automatically.

  All variants must share the same kinematic structure (same bodies,
  joints, joint types). Only mesh geoms can differ.

  Variant assignment is fixed at ``Simulation`` initialization; it does
  not resample on episode reset.
  """

  variants: dict[str, Callable[[], mujoco.MjSpec]] = field(default_factory=dict)
  """Named mesh variants. Each value is a callable returning an ``MjSpec``."""

  assignment: dict[str, float] | Callable[[int], Sequence[int]] | None = None
  """How worlds get mapped to variants. Three shapes:

  * ``None`` (default): uniform allocation across variants via
    largest-remainder.
  * ``dict[str, float]``: per-variant weights for largest-remainder
    allocation. Variants not listed default to weight 1.0.
  * ``Callable[[int], Sequence[int]]``: an explicit assignment
    function, called with ``num_envs`` at simulation init and required
    to return a length-``num_envs`` sequence of variant indices in
    ``[0, len(variants))``."""

  spec_fn: Callable[[], mujoco.MjSpec] = field(default=_variant_spec_fn_unset)
  """Unused on ``VariantEntityCfg``; the merged spec is built from ``variants``."""

  def __post_init__(self) -> None:
    if self.spec_fn is not _variant_spec_fn_unset:
      raise ValueError(
        "VariantEntityCfg.spec_fn cannot be set; pass per-variant spec "
        "callables in `variants` instead."
      )
    if isinstance(self.assignment, dict):
      extras = set(self.assignment) - set(self.variants)
      if extras:
        raise ValueError(
          f"VariantEntityCfg.assignment has keys not present in variants: "
          f"{sorted(extras)}."
        )


##
# Output: data records that flow to scene/sim.
##


@dataclass(frozen=True)
class SlotKey:
  """Structural identity of a mesh slot in the merged template."""

  body_path: str
  role: GeomRole
  ordinal: int


@dataclass(frozen=True)
class VariantGeomSpec:
  """Per-variant per-slot geom attributes captured from the source spec.

  The slot-aware merge uses this to restore exact source semantics
  (contact bits, friction, material, etc.) for each variant's reference
  compilation, instead of inheriting whatever the template variant set.
  """

  mesh_name: str
  geom_name: str | None
  contype: int
  conaffinity: int
  condim: int
  group: int
  priority: int
  material: str | None
  rgba: tuple[float, float, float, float]
  friction: tuple[float, float, float]
  margin: float
  gap: float
  solref: tuple[float, float]
  solimp: tuple[float, float, float, float, float]
  # ``mass``/``density`` default to 0.0 in the source spec, which MuJoCo
  # interprets as "infer from mesh volume". The slot-aware compile path
  # passes the source value through verbatim.
  mass: float
  density: float


@dataclass(frozen=True)
class VariantSlot:
  """One slot in the merged template body.

  ``template_geom_name`` is the synthesized name on the template body's
  geom that backs this slot. It always uses the reserved
  ``mjlab/pad/<body_path>/<role>/<ordinal>`` prefix so it cannot
  collide with user-named source geoms (validation rejects user geoms
  starting with ``mjlab/pad/``).

  ``source_geom_names`` records, per variant, the original source geom
  name for diagnostics. ``None`` here can mean either "variant doesn't
  fill the slot" or "variant fills the slot with an unnamed geom"; the
  authoritative source for fill status is
  ``VariantMetadata.variant_slot_specs[v][s] is None``.
  """

  key: SlotKey
  template_geom_name: str
  source_geom_names: tuple[str | None, ...]


@dataclass
class VariantMetadata:
  """Bookkeeping produced when merging variant specs.

  Slots are ordered by ``(body_path, role, ordinal)`` and
  ``variant_slot_specs[v]`` aligns positionally with ``slots`` so
  ``variant_slot_specs[v][s]`` describes how variant ``v`` fills slot
  ``s`` (``None`` if variant ``v`` leaves the slot unfilled).

  ``variant_source_specs[v]`` is the original ``MjSpec`` produced by
  variant ``v``'s ``spec_fn`` (pre-merge). The sim-time build path
  compiles each one in isolation rather than recompiling a copy of the
  merged scene, which keeps construction cost linear in the number of
  variants instead of quadratic.

  ``variant_mesh_names`` and ``num_mesh_geoms`` are kept as derived
  ``@property`` views over ``variant_slot_specs`` for back-compat with
  consumers that index by slot.
  """

  variant_names: tuple[str, ...]
  assignment: tuple[float, ...] | Callable[[int], Sequence[int]] = ()
  slots: tuple[VariantSlot, ...] = ()
  variant_slot_specs: tuple[tuple[VariantGeomSpec | None, ...], ...] = ()
  variant_source_specs: tuple[mujoco.MjSpec, ...] = ()

  @property
  def variant_mesh_names(self) -> tuple[tuple[str | None, ...], ...]:
    """Per-variant slot-aligned mesh names with variant prefix.

    ``None`` at position ``s`` means variant ``v`` does not fill slot
    ``s``. Derived from ``variant_slot_specs``.
    """
    return tuple(
      tuple(
        None if ss is None else f"{self.variant_names[v_idx]}/{ss.mesh_name}"
        for ss in slot_specs
      )
      for v_idx, slot_specs in enumerate(self.variant_slot_specs)
    )

  @property
  def num_mesh_geoms(self) -> int:
    """Max number of filled mesh slots across variants (i.e. the longest
    variant's source mesh count). Derived from ``variant_slot_specs``."""
    if not self.variant_slot_specs:
      return 0
    return max(
      sum(1 for ss in slot_specs if ss is not None)
      for slot_specs in self.variant_slot_specs
    )


@dataclass
class MeshVariantResult:
  """Output of :func:`build_variant_model`."""

  wp_model: mjwarp.Model
  mj_model: mujoco.MjModel
  # Maps entity prefix -> array of variant indices per world.
  world_to_variant: dict[str, np.ndarray]


##
# Tree helpers (shared by validation and slot computation).
##


def _iter_body_paths(
  body: mujoco.MjsBody, parent_path: str = ""
) -> list[tuple[str, mujoco.MjsBody]]:
  """Return ``[(path, body), ...]`` for the recursive body tree.

  Path is the body's path relative to the variant root (e.g. ``/prop``,
  ``/prop/lid``), built by joining body names with ``/``.
  """
  body_name = body.name or ""
  path = f"{parent_path}/{body_name}"
  out: list[tuple[str, mujoco.MjsBody]] = [(path, body)]
  for child in body.bodies:
    out.extend(_iter_body_paths(child, path))
  return out


def _classify_geom_role(geom: mujoco.MjsGeom) -> GeomRole:
  """Derive a geom's slot role from its contact bits.

  Visual: ``contype == 0`` and ``conaffinity == 0``. Collision: anything
  else. Shared by validation and slot computation.
  """
  if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
    return "visual"
  return "collision"


def _iter_body_tree(body: mujoco.MjsBody):
  yield body
  for child in body.bodies:
    yield from _iter_body_tree(child)


##
# Validation pass: cross-variant topology comparison.
##


@dataclass(frozen=True)
class JointSignature:
  """Joint identity for cross-variant matching.

  Captures the structural attributes that must match across variants:
  name, type, and qpos/qvel widths. Parameter attributes (axis, range,
  stiffness, damping, etc.) are not part of the structural signature;
  they may legitimately differ across variants once registered as
  per-world fields.
  """

  name: str
  type: int
  qpos_width: int
  qvel_width: int


@dataclass(frozen=True)
class GeomSignature:
  """Geom identity for cross-variant matching.

  ``role`` is derived from ``contype``/``conaffinity``: a geom that
  participates in any contact pair is "collision"; otherwise "visual".
  A slot's role is fixed across variants by construction because
  mujoco_warp's ``geom_contype``/``geom_conaffinity`` are 1D shared.
  """

  name: str | None
  type: int
  role: GeomRole


@dataclass(frozen=True)
class BodySignature:
  """Recursive body identity for cross-variant matching.

  ``path`` is the body path relative to its variant's worldbody, before
  any scene-attachment prefix. ``children`` are nested signatures in
  source order.
  """

  path: str
  name: str
  joints: tuple[JointSignature, ...]
  geoms: tuple[GeomSignature, ...]
  children: tuple["BodySignature", ...]


def _format_variant_error(
  variant_name: str, message: str, hint: str | None = None
) -> str:
  prefix = f"mjlab.entity: VariantEntityCfg '{variant_name}': "
  body = message
  if hint:
    body += f" Hint: {hint}"
  return prefix + body


def _extract_body_signature(
  body: mujoco.MjsBody, parent_path: str = ""
) -> BodySignature:
  body_name = body.name or ""
  path = f"{parent_path}/{body_name}"
  joints: list[JointSignature] = []
  for j in body.joints:
    jt = int(j.type)
    joints.append(
      JointSignature(
        name=j.name or "",
        type=jt,
        qpos_width=qpos_width(jt),
        qvel_width=dof_width(jt),
      )
    )
  geoms = tuple(
    GeomSignature(
      name=g.name if g.name else None,
      type=int(g.type),
      role=_classify_geom_role(g),
    )
    for g in body.geoms
  )
  children = tuple(_extract_body_signature(child, path) for child in body.bodies)
  return BodySignature(
    path=path,
    name=body_name,
    joints=tuple(joints),
    geoms=geoms,
    children=children,
  )


def _detect_inertial_mode(body: mujoco.MjsBody) -> InertialMode:
  # `body.fullinertia` defaults to [nan, 0, 0, 0, 0, 0]; a non-NaN first
  # element means the user assigned fullinertia (any assignment, even
  # zeros, flips the user-specified flag at compile time).
  if not math.isnan(float(body.fullinertia[0])):
    return "fullinertia"
  if int(body.explicitinertial):
    return "diagonal"
  return "mesh-derived"


def _collect_inertial_modes(
  body: mujoco.MjsBody, parent_path: str = ""
) -> dict[str, InertialMode]:
  body_name = body.name or ""
  path = f"{parent_path}/{body_name}"
  modes: dict[str, InertialMode] = {path: _detect_inertial_mode(body)}
  for child in body.bodies:
    modes.update(_collect_inertial_modes(child, path))
  return modes


def _check_reserved_names(spec: mujoco.MjSpec, variant_name: str) -> None:
  collections: tuple[tuple[str, str], ...] = (
    ("bodies", "body"),
    ("geoms", "geom"),
    ("meshes", "mesh"),
    ("materials", "material"),
    ("joints", "joint"),
    ("actuators", "actuator"),
    ("tendons", "tendon"),
    ("sensors", "sensor"),
    ("cameras", "camera"),
    ("lights", "light"),
    ("sites", "site"),
    ("equalities", "equality"),
  )
  for attr, item_kind in collections:
    items = getattr(spec, attr, None)
    if items is None:
      continue
    for item in items:
      name = getattr(item, "name", None) or ""
      for prefix in RESERVED_NAME_PREFIXES:
        if name.startswith(prefix):
          raise ValueError(
            _format_variant_error(
              variant_name,
              f"reserved name prefix '{prefix}' is used by source "
              f"{item_kind} '{name}'.",
              "rename user geoms/bodies/assets away from this prefix.",
            )
          )


def _compare_geoms_for_body(
  ref_variant: str,
  ref_geoms: tuple[GeomSignature, ...],
  other_variant: str,
  other_geoms: tuple[GeomSignature, ...],
  body_path: str,
) -> None:
  """Compare geoms within a matching body across two variants.

  Mesh geoms may differ in count per ``(body, role)``; they form padded
  slots and are not part of the structural signature. Non-mesh
  primitive geoms must match exactly in count, name, type, role, and
  ordering relative to other primitives.
  """
  mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
  ref_primitives = tuple(g for g in ref_geoms if g.type != mesh_type)
  other_primitives = tuple(g for g in other_geoms if g.type != mesh_type)
  if len(ref_primitives) != len(other_primitives):
    raise ValueError(
      _format_variant_error(
        other_variant,
        f"body '{body_path}' has {len(other_primitives)} non-mesh "
        f"geoms, but '{ref_variant}' has {len(ref_primitives)}.",
        "primitive geoms must match across all variants; only mesh "
        "geom counts may differ.",
      )
    )
  for i, (rp, op) in enumerate(zip(ref_primitives, other_primitives, strict=False)):
    if rp != op:
      raise ValueError(
        _format_variant_error(
          other_variant,
          f"body '{body_path}' primitive geom #{i} differs from "
          f"'{ref_variant}': got {op}, expected {rp}.",
          "primitive geom name, type, and role must match across variants.",
        )
      )


def _compare_body_signatures(
  ref_variant: str,
  ref: BodySignature,
  other_variant: str,
  other: BodySignature,
) -> None:
  if ref.path != other.path:
    raise ValueError(
      _format_variant_error(
        other_variant,
        f"body path '{other.path}' has no counterpart in "
        f"'{ref_variant}' (expected '{ref.path}').",
        "variants must share the same recursive body tree with matching body names.",
      )
    )
  if len(ref.joints) != len(other.joints):
    raise ValueError(
      _format_variant_error(
        other_variant,
        f"body '{other.path}' has {len(other.joints)} joints, but "
        f"'{ref_variant}' has {len(ref.joints)}.",
        "variants may change mesh assets, not add/remove joints.",
      )
    )
  for i, (rj, oj) in enumerate(zip(ref.joints, other.joints, strict=False)):
    if rj != oj:
      raise ValueError(
        _format_variant_error(
          other_variant,
          f"body '{other.path}' joint #{i} differs from "
          f"'{ref_variant}': got {oj}, expected {rj}.",
          "joint name, type, and qpos/qvel width must match across variants.",
        )
      )
  _compare_geoms_for_body(
    ref_variant, ref.geoms, other_variant, other.geoms, other.path
  )
  if len(ref.children) != len(other.children):
    raise ValueError(
      _format_variant_error(
        other_variant,
        f"body '{other.path}' has {len(other.children)} child bodies, "
        f"but '{ref_variant}' has {len(ref.children)}.",
        "variants must share the same recursive body tree.",
      )
    )
  for ref_child, other_child in zip(ref.children, other.children, strict=False):
    _compare_body_signatures(ref_variant, ref_child, other_variant, other_child)


def _compare_inertial_modes(
  ref_variant: str,
  ref_modes: dict[str, InertialMode],
  other_variant: str,
  other_modes: dict[str, InertialMode],
) -> None:
  for path in sorted(set(ref_modes) | set(other_modes)):
    rm = ref_modes.get(path)
    om = other_modes.get(path)
    if rm is None or om is None:
      # Path mismatch is reported by the body-tree comparison.
      continue
    if rm != om:
      raise ValueError(
        _format_variant_error(
          other_variant,
          f"body '{path}' uses inertial representation '{om}', but "
          f"'{ref_variant}' uses '{rm}'.",
          "use the same inertial representation in every variant: "
          "all diagonal, all fullinertia, or all mesh-derived.",
        )
      )


def _compare_collection_topology(
  *,
  item_kind: str,
  ref_variant: str,
  ref_items: list[Any],
  other_variant: str,
  other_items: list[Any],
  key_fn: Callable[[Any], tuple],
  count_hint: str,
  item_hint: str,
) -> None:
  """Compare a flat spec-level collection (actuators, sensors, ...).

  Length must match; per-element ``key_fn(item)`` tuples must match.
  Per-collection error specificity is preserved through ``item_kind``,
  ``count_hint``, and ``item_hint`` strings supplied by the caller.
  """
  if len(ref_items) != len(other_items):
    raise ValueError(
      _format_variant_error(
        other_variant,
        f"{item_kind} count {len(other_items)} differs from "
        f"'{ref_variant}' ({len(ref_items)}).",
        count_hint,
      )
    )
  for i, (r, o) in enumerate(zip(ref_items, other_items, strict=False)):
    rk, ok = key_fn(r), key_fn(o)
    if rk != ok:
      raise ValueError(
        _format_variant_error(
          other_variant,
          f"{item_kind} #{i} differs from '{ref_variant}': got {ok}, expected {rk}.",
          item_hint,
        )
      )


def _actuator_key(a: mujoco.MjsActuator) -> tuple[str, int, str]:
  return (a.name or "", int(a.trntype), a.target or "")


def _sensor_key(s: mujoco.MjsSensor) -> tuple[str, int, int, str]:
  return (s.name or "", int(s.type), int(s.objtype), s.objname or "")


def _tendon_key(t: mujoco.MjsTendon) -> tuple[str]:
  return (t.name or "",)


def _equality_key(e: mujoco.MjsEquality) -> tuple[str, int, str, str]:
  return (e.name or "", int(e.type), e.name1 or "", e.name2 or "")


def validate_variant_specs(
  names: list[str],
  specs: list[mujoco.MjSpec],
) -> None:
  """Validate that all variant specs share the same kinematic topology.

  Performs recursive body-tree, joint, and primitive-geom comparison;
  spec-level actuator/sensor/tendon/equality topology comparison;
  per-body inertial-mode consistency; and reserved-name-prefix
  rejection. Raises ``ValueError`` with a message of the form:

      mjlab.entity: VariantEntityCfg '<name>': <reason>. Hint: <fix>.

  Mesh geom counts may differ per ``(body, role)``; this is the
  intended source of variant-to-variant variation. Primitive (non-mesh)
  geom counts and structure must match.

  Args:
    names: Variant names. Must be non-empty and align with ``specs``.
    specs: Variant specs (pre-attachment). Each must have exactly one
      root body under worldbody.
  """
  if len(names) != len(specs):
    raise ValueError("names and specs must have the same length.")
  if not names:
    raise ValueError("at least one variant is required.")

  for name, spec in zip(names, specs, strict=False):
    _check_reserved_names(spec, name)

  root_bodies: list[mujoco.MjsBody] = []
  for name, spec in zip(names, specs, strict=False):
    children = list(spec.worldbody.bodies)
    if len(children) != 1:
      raise ValueError(
        _format_variant_error(
          name,
          f"variant must have exactly one root body under worldbody, "
          f"got {len(children)}.",
          "place the variant's root body directly under worldbody.",
        )
      )
    root_bodies.append(children[0])

  ref_name = names[0]
  ref_signature = _extract_body_signature(root_bodies[0])
  ref_modes = _collect_inertial_modes(root_bodies[0])
  ref_spec = specs[0]

  for i in range(1, len(names)):
    other_name = names[i]
    other_signature = _extract_body_signature(root_bodies[i])
    other_modes = _collect_inertial_modes(root_bodies[i])
    other_spec = specs[i]

    _compare_body_signatures(ref_name, ref_signature, other_name, other_signature)
    _compare_inertial_modes(ref_name, ref_modes, other_name, other_modes)
    _compare_collection_topology(
      item_kind="actuator",
      ref_variant=ref_name,
      ref_items=list(ref_spec.actuators),
      other_variant=other_name,
      other_items=list(other_spec.actuators),
      key_fn=_actuator_key,
      count_hint="keep control and observation dimensions fixed across variants.",
      item_hint=(
        "actuator topology (name, transmission type, target) must match "
        "across variants."
      ),
    )
    _compare_collection_topology(
      item_kind="sensor",
      ref_variant=ref_name,
      ref_items=list(ref_spec.sensors),
      other_variant=other_name,
      other_items=list(other_spec.sensors),
      key_fn=_sensor_key,
      count_hint="keep sensor topology identical across variants.",
      item_hint="sensor topology (name, type, target object) must match.",
    )
    _compare_collection_topology(
      item_kind="tendon",
      ref_variant=ref_name,
      ref_items=list(ref_spec.tendons),
      other_variant=other_name,
      other_items=list(other_spec.tendons),
      key_fn=_tendon_key,
      count_hint="keep tendon topology identical across variants.",
      item_hint="tendon names must match across variants.",
    )
    _compare_collection_topology(
      item_kind="equality",
      ref_variant=ref_name,
      ref_items=list(ref_spec.equalities),
      other_variant=other_name,
      other_items=list(other_spec.equalities),
      key_fn=_equality_key,
      count_hint="keep equality constraints identical across variants.",
      item_hint="equality constraint topology must match across variants.",
    )


##
# Slot computation: assign (body_path, role, ordinal) keys to mesh geoms.
##


def _extract_variant_geom_spec(g: mujoco.MjsGeom) -> VariantGeomSpec:
  """Snapshot a source mesh geom's attributes for slot-aware compile."""
  return VariantGeomSpec(
    mesh_name=g.meshname,
    geom_name=g.name if g.name else None,
    contype=int(g.contype),
    conaffinity=int(g.conaffinity),
    condim=int(g.condim),
    group=int(g.group),
    priority=int(g.priority),
    material=g.material if g.material else None,
    rgba=(
      float(g.rgba[0]),
      float(g.rgba[1]),
      float(g.rgba[2]),
      float(g.rgba[3]),
    ),
    friction=(
      float(g.friction[0]),
      float(g.friction[1]),
      float(g.friction[2]),
    ),
    margin=float(g.margin),
    gap=float(g.gap),
    solref=(float(g.solref[0]), float(g.solref[1])),
    solimp=(
      float(g.solimp[0]),
      float(g.solimp[1]),
      float(g.solimp[2]),
      float(g.solimp[3]),
      float(g.solimp[4]),
    ),
    mass=float(g.mass),
    density=float(g.density),
  )


def _compute_slot_metadata(
  variant_names: tuple[str, ...],
  variant_specs: list[mujoco.MjSpec],
) -> tuple[
  tuple[VariantSlot, ...],
  tuple[tuple[VariantGeomSpec | None, ...], ...],
]:
  """Compute slot list and per-variant slot specs.

  Walks each variant's body tree, buckets mesh geoms by ``(body_path,
  role)``, and assigns ordinals deterministically from source order.
  Slot count per ``(body_path, role)`` equals the max across variants;
  variants with fewer geoms in that bucket leave trailing slots
  unfilled (``None`` in ``variant_slot_specs``).
  """
  mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)

  # Per-variant {body_path -> {role -> [VariantGeomSpec]}}.
  per_variant_buckets: list[dict[str, dict[GeomRole, list[VariantGeomSpec]]]] = []
  for spec in variant_specs:
    root = list(spec.worldbody.bodies)[0]
    buckets: dict[str, dict[GeomRole, list[VariantGeomSpec]]] = {}
    for path, body in _iter_body_paths(root):
      role_buckets = buckets.setdefault(path, {"visual": [], "collision": []})
      for g in body.geoms:
        if int(g.type) != mesh_type:
          continue
        role_buckets[_classify_geom_role(g)].append(_extract_variant_geom_spec(g))
    per_variant_buckets.append(buckets)

  # All body paths present in any variant. (Validation ensures the path
  # sets agree across variants, but use the union here for safety.)
  all_paths: set[str] = set()
  for buckets in per_variant_buckets:
    all_paths.update(buckets.keys())

  slots: list[VariantSlot] = []
  per_variant_slot_specs: list[list[VariantGeomSpec | None]] = [
    [] for _ in variant_names
  ]

  # Deterministic order: sort body paths lexicographically, then visual
  # before collision per body, then by ordinal.
  for body_path in sorted(all_paths):
    for role in ("visual", "collision"):
      max_count = 0
      for buckets in per_variant_buckets:
        count = len(buckets.get(body_path, {}).get(role, []))
        if count > max_count:
          max_count = count
      for ordinal in range(max_count):
        source_names: list[str | None] = []
        slot_specs: list[VariantGeomSpec | None] = []
        for variant_buckets in per_variant_buckets:
          variant_geoms = variant_buckets.get(body_path, {}).get(role, [])
          if ordinal < len(variant_geoms):
            geom_spec = variant_geoms[ordinal]
            source_names.append(geom_spec.geom_name)
            slot_specs.append(geom_spec)
          else:
            source_names.append(None)
            slot_specs.append(None)
        template_name = f"mjlab/pad{body_path}/{role}/{ordinal}"
        slots.append(
          VariantSlot(
            key=SlotKey(
              body_path=body_path,
              role=role,
              ordinal=ordinal,
            ),
            template_geom_name=template_name,
            source_geom_names=tuple(source_names),
          )
        )
        for variant_idx, slot_spec in enumerate(slot_specs):
          per_variant_slot_specs[variant_idx].append(slot_spec)

  return tuple(slots), tuple(tuple(s) for s in per_variant_slot_specs)


##
# Build pipeline (entity-time): merge variant specs into one padded template.
##


def build_merged_variant_spec(
  cfg: VariantEntityCfg,
) -> tuple[mujoco.MjSpec, VariantMetadata]:
  """Merge a ``VariantEntityCfg``'s variants into a single padded template.

  Validates that all variants share the same kinematic structure, merges
  every variant's mesh assets into the template namespace, slot-renames
  variant 0's mesh geoms to ``mjlab/pad/<body>/<role>/<ordinal>``, and
  synthesizes padding geoms (per-(body, role)) for slots that variant 0
  does not fill, so the template's geom count and bodyid layout cover
  every variant's needs.

  Returns the merged template spec and a ``VariantMetadata`` describing
  the slot layout, per-variant slot specs, and explicit body inertials.
  The reference compile in :func:`build_variant_model` overrides every
  per-variant attribute via ``VariantGeomSpec``, so variant 0 being "the
  template" does not privilege its attributes.
  """
  variants = cfg.variants
  if not variants:
    raise ValueError("VariantEntityCfg.variants must contain at least one entry.")

  variant_names: list[str] = []
  variant_specs: list[mujoco.MjSpec] = []
  for name, spec_fn in variants.items():
    variant_names.append(name)
    variant_specs.append(spec_fn())

  # Resolve cfg.assignment (None | dict | callable) into the metadata's
  # unified ``assignment`` field. None and dict both produce a weights
  # tuple keyed by variant declaration order; a callable passes through.
  resolved_assignment: tuple[float, ...] | Callable[[int], Sequence[int]]
  cfg_assignment = cfg.assignment
  if cfg_assignment is None:
    resolved_assignment = (1.0,) * len(variant_names)
  elif isinstance(cfg_assignment, dict):
    weights_dict = cast(dict[str, float], cfg_assignment)
    resolved_assignment = tuple(weights_dict.get(n, 1.0) for n in variant_names)
  else:
    resolved_assignment = cfg_assignment

  # Validate cross-variant topology (recursive body tree, joints,
  # primitive geoms, actuators, sensors, tendons, equalities, inertial
  # mode consistency, and reserved-prefix collisions). The validator
  # enforces "exactly one root body under worldbody" itself.
  validate_variant_specs(variant_names, variant_specs)

  variant_bodies: list[mujoco.MjsBody] = [
    list(spec.worldbody.bodies)[0] for spec in variant_specs
  ]

  # Variant entities must be floating-base. Mocap auto-wrap is not applied
  # for variant entities, so fixed-base variants would silently stack at
  # the world origin. Variants share joint structure (validated above), so
  # checking the first is sufficient.
  ref_joints = list(variant_bodies[0].joints)
  if not ref_joints or ref_joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
    raise ValueError(
      "VariantEntityCfg requires floating-base variants. Each variant's "
      "root body must declare a free joint via body.add_freejoint(); "
      "fixed-base variants are not supported."
    )

  # Compute slot-based metadata from source specs BEFORE any mesh
  # renaming or merging. This captures (body_path, role, ordinal) keyed
  # slots and per-variant geom attributes for the slot-aware build path.
  slots, variant_slot_specs = _compute_slot_metadata(
    tuple(variant_names), variant_specs
  )

  # Snapshot a clean copy of every source spec BEFORE the merge mutates
  # variant 0's spec (mesh renames, slot-name geom renames, padding geom
  # synthesis). The sim-time build compiles each of these in isolation
  # to populate per-world fields, avoiding the O(N) cost of recompiling
  # the merged scene per unique variant assignment.
  variant_source_specs = tuple(s.copy() for s in variant_specs)

  # Use variant 0's spec as the template. The merge:
  #   1. prefixes variant 0's mesh names with its variant name and
  #      copies every other variant's mesh assets into the template;
  #   2. renames variant 0's mesh geoms to slot template names so
  #      every mesh geom in the merged template is addressable by
  #      ``slot.template_geom_name``;
  #   3. synthesizes padding geoms (per-(body, role)) for slots that
  #      variant 0 does not fill, so the template's geom count and
  #      bodyid layout cover every variant's needs.
  template_spec = variant_specs[0]
  template_body = variant_bodies[0]

  # (1) Rename template meshes and update mesh references on the
  # template body's geoms (recursive: child bodies may also reference
  # the template variant's meshes).
  template_prefix = f"{variant_names[0]}/"
  old_to_new: dict[str, str] = {}
  for mesh in template_spec.meshes:
    new_name = f"{template_prefix}{mesh.name}"
    old_to_new[mesh.name] = new_name
    mesh.name = new_name
  for _, body in _iter_body_paths(template_body):
    for g in body.geoms:
      if g.meshname in old_to_new:
        g.meshname = old_to_new[g.meshname]

  # Copy mesh assets from other variants into the template namespace.
  for i in range(1, len(variant_specs)):
    prefix = f"{variant_names[i]}/"
    for mesh in variant_specs[i].meshes:
      new_mesh = template_spec.add_mesh()
      new_mesh.name = f"{prefix}{mesh.name}"
      copy_mesh_data(mesh, new_mesh)

  # (2) Slot-driven rename of variant 0's existing mesh geoms. Walk
  # the template body tree; within each (body, role) bucket, the
  # n-th mesh geom (in source order) maps to slot ordinal n.
  mesh_type = mujoco.mjtGeom.mjGEOM_MESH
  slot_by_key: dict[SlotKey, VariantSlot] = {s.key: s for s in slots}
  template_bodies_by_path: dict[str, mujoco.MjsBody] = {}
  filled_template_slot_keys: set[SlotKey] = set()
  for body_path, body in _iter_body_paths(template_body):
    template_bodies_by_path[body_path] = body
    role_ordinals: dict[GeomRole, int] = {"visual": 0, "collision": 0}
    for g in body.geoms:
      if g.type != mesh_type:
        continue
      role = _classify_geom_role(g)
      ordinal = role_ordinals[role]
      role_ordinals[role] += 1
      key = SlotKey(body_path=body_path, role=role, ordinal=ordinal)
      slot = slot_by_key[key]
      g.name = slot.template_geom_name
      filled_template_slot_keys.add(key)

  # (3) Synthesize per-(body, role) padding geoms for slots variant 0
  # doesn't fill. Each padding geom uses the slot's reserved
  # ``mjlab/pad/...`` template name, the role's union contact bits,
  # and a placeholder mesh from the lowest-index variant that fills
  # the slot. ``mass=0`` and ``density=0`` keep the placeholder from
  # contributing to the template's body inertial inference; the
  # per-variant reference compile handles further masking.
  for s_idx, slot in enumerate(slots):
    if slot.key in filled_template_slot_keys:
      continue
    target_body = template_bodies_by_path[slot.key.body_path]
    pad_geom = target_body.add_geom()
    pad_geom.type = mesh_type
    pad_geom.name = slot.template_geom_name
    if slot.key.role == "visual":
      pad_geom.contype = 0
      pad_geom.conaffinity = 0
    else:
      pad_geom.contype = 1
      pad_geom.conaffinity = 1
    placeholder_set = False
    for v_idx, slot_specs in enumerate(variant_slot_specs):
      gs = slot_specs[s_idx]
      if gs is None:
        continue
      pad_geom.meshname = f"{variant_names[v_idx]}/{gs.mesh_name}"
      placeholder_set = True
      break
    assert placeholder_set, (
      f"slot {slot.key} has no filling variant; "
      "_compute_slot_metadata must not emit unfilled slots."
    )
    pad_geom.mass = 0.0
    pad_geom.density = 0.0

  metadata = VariantMetadata(
    variant_names=tuple(variant_names),
    assignment=resolved_assignment,
    slots=slots,
    variant_slot_specs=variant_slot_specs,
    variant_source_specs=variant_source_specs,
  )
  return template_spec, metadata


##
# Build pipeline (sim-time): produce a heterogeneous Warp model.
#
# The module-level entry point is ``build_variant_model`` at the bottom
# of this section. ``allocate_worlds`` is its world-assignment step and
# is also exposed for unit testing the largest-remainder math in
# isolation. Everything between is private to ``build_variant_model``
# (or to ``_populate_dependent_fields``, which is itself private).
##


def _qualified_mesh_name(entity_prefix: str, variant_name: str, mesh_name: str) -> str:
  """Mesh asset name in the merged template's namespace.

  Source mesh names are prefixed with the variant name during merge
  (``<variant>/<mesh>``); after scene attach the entity prefix is
  added (``<entity>/<variant>/<mesh>``).
  """
  return f"{entity_prefix}{variant_name}/{mesh_name}"


def _qualified_slot_geom_name(entity_prefix: str, template_geom_name: str) -> str:
  """Slot geom name in the merged template's namespace after attach."""
  return f"{entity_prefix}{template_geom_name}"


def _populate_dependent_fields(
  m: mjwarp.Model,
  padded_model: mujoco.MjModel,
  nworld: int,
  variant_info: list[tuple[str, VariantMetadata]],
  world_to_variant: dict[str, np.ndarray],
) -> None:
  """Populate per-world Warp arrays for the geometry-dependent fields.

  Compiles each variant's *source* spec in isolation (small, single-entity
  spec) and scatters per-body and per-geom fields into per-world arrays.
  Total cost is the sum of per-source compiles across all variant
  entities, i.e. linear in the number of variants. This avoids the
  O(N^2) cost of recompiling the merged scene per unique variant
  assignment.

  For non-variant bodies and non-slot geoms, values come from a single
  base scene compile (``padded_model``) broadcast across worlds. For
  variant entity bodies and slot geoms, the per-world value comes from
  the corresponding variant's source compile. ``body_subtreemass`` for
  ancestors of each variant entity's root body is updated per-world by
  applying the per-variant subtreemass delta.
  """
  ngeom = padded_model.ngeom
  nbody = padded_model.nbody

  # Initialize per-world arrays from the base scene compile. Bodies and
  # geoms not touched by any variant entity inherit these values. We
  # broadcast then copy so writes to per-world rows don't alias.
  def _broadcast(src: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(src.astype(np.float32), shape).copy()

  geom_size = _broadcast(padded_model.geom_size, (nworld, ngeom, 3))
  geom_rbound = _broadcast(padded_model.geom_rbound, (nworld, ngeom))
  geom_aabb = _broadcast(
    padded_model.geom_aabb.reshape(ngeom, 2, 3), (nworld, ngeom, 2, 3)
  )
  geom_pos = _broadcast(padded_model.geom_pos, (nworld, ngeom, 3))
  geom_quat = _broadcast(padded_model.geom_quat, (nworld, ngeom, 4))
  body_mass = _broadcast(padded_model.body_mass, (nworld, nbody))
  body_subtreemass = _broadcast(padded_model.body_subtreemass, (nworld, nbody))
  body_inertia = _broadcast(padded_model.body_inertia, (nworld, nbody, 3))
  body_invweight0 = _broadcast(padded_model.body_invweight0, (nworld, nbody, 2))
  body_ipos = _broadcast(padded_model.body_ipos, (nworld, nbody, 3))
  body_iquat = _broadcast(padded_model.body_iquat, (nworld, nbody, 4))

  mesh_type = mujoco.mjtGeom.mjGEOM_MESH

  for entity_prefix, metadata in variant_info:
    w2v = world_to_variant[entity_prefix]

    # Use any variant's source spec to enumerate body paths and the
    # entity's slot geom names. Validation guarantees variants share
    # the same body tree and slot keys, so the choice is arbitrary.
    sample_spec = metadata.variant_source_specs[0]
    sample_root = list(sample_spec.worldbody.bodies)[0]

    # Map each variant-entity body path to its scene body id. Body
    # names in the scene are the bare body name prefixed by the
    # entity's attach prefix.
    scene_body_id_by_path: dict[str, int] = {}
    body_name_by_path: dict[str, str] = {}
    for body_path, body in _iter_body_paths(sample_root):
      body_name_by_path[body_path] = body.name or ""
      scene_name = f"{entity_prefix}{body.name or ''}"
      bid = mujoco.mj_name2id(padded_model, mujoco.mjtObj.mjOBJ_BODY, scene_name)
      if bid < 0:
        raise ValueError(
          f"variant entity body '{scene_name}' not found in compiled scene."
        )
      scene_body_id_by_path[body_path] = bid

    # Map each slot key to its scene geom id.
    scene_geom_id_by_slot: dict[SlotKey, int] = {}
    for slot in metadata.slots:
      full_geom_name = _qualified_slot_geom_name(entity_prefix, slot.template_geom_name)
      gid = mujoco.mj_name2id(padded_model, mujoco.mjtObj.mjOBJ_GEOM, full_geom_name)
      if gid < 0:
        raise ValueError(
          f"slot geom '{full_geom_name}' (entity '{entity_prefix}') not "
          f"found in compiled scene."
        )
      scene_geom_id_by_slot[slot.key] = gid

    # Variant entity's root body (shortest body path), used to compute
    # the subtreemass delta to propagate to ancestors. The root path is
    # always the lex-smallest among the entity's body paths (parents
    # sort before children).
    sorted_paths = sorted(scene_body_id_by_path.keys())
    root_body_path = sorted_paths[0]
    root_scene_bid = scene_body_id_by_path[root_body_path]

    # Walk up the scene's parent chain from the variant root to
    # worldbody (inclusive). The variant root's own subtreemass is
    # written from the per-variant compile; each ancestor receives an
    # additive delta equal to (variant_subtreemass - base_subtreemass).
    ancestor_ids: list[int] = []
    parent_id = int(padded_model.body_parentid[root_scene_bid])
    while True:
      ancestor_ids.append(parent_id)
      if parent_id == 0:
        break
      parent_id = int(padded_model.body_parentid[parent_id])
    base_root_subtreemass = float(padded_model.body_subtreemass[root_scene_bid])

    # Compile each variant's source spec in isolation and scatter values
    # into the per-world arrays for the worlds assigned to that variant.
    for v_idx in range(len(metadata.variant_names)):
      worlds = np.where(w2v == v_idx)[0]
      if worlds.size == 0:
        continue

      source_spec = metadata.variant_source_specs[v_idx].copy()
      v_model = source_spec.compile()

      source_root = list(source_spec.worldbody.bodies)[0]
      source_body_id_by_path: dict[str, int] = {}
      for body_path, body in _iter_body_paths(source_root):
        source_body_id_by_path[body_path] = body.id

      # Scatter per-body fields for every body in the variant entity's
      # subtree. Bodies without explicit inertials inherit the
      # mesh-derived inertia from the variant's mesh assignment in the
      # source compile, which is exactly what we want.
      for body_path, scene_bid in scene_body_id_by_path.items():
        source_bid = source_body_id_by_path[body_path]
        body_mass[worlds, scene_bid] = v_model.body_mass[source_bid]
        body_subtreemass[worlds, scene_bid] = v_model.body_subtreemass[source_bid]
        body_inertia[worlds, scene_bid] = v_model.body_inertia[source_bid]
        body_invweight0[worlds, scene_bid] = v_model.body_invweight0[source_bid]
        body_ipos[worlds, scene_bid] = v_model.body_ipos[source_bid]
        body_iquat[worlds, scene_bid] = v_model.body_iquat[source_bid]

      # Scatter per-geom fields for each slot this variant fills. Map
      # each (body_path, role, ordinal) slot to the corresponding geom
      # in the source compile by re-walking the source body tree (same
      # ordering rules as ``_compute_slot_metadata``).
      for body_path, body in _iter_body_paths(source_root):
        role_ordinals: dict[GeomRole, int] = {"visual": 0, "collision": 0}
        for g in body.geoms:
          if g.type != mesh_type:
            continue
          role = _classify_geom_role(g)
          ordinal = role_ordinals[role]
          role_ordinals[role] += 1
          slot_key = SlotKey(body_path=body_path, role=role, ordinal=ordinal)
          scene_gid = scene_geom_id_by_slot.get(slot_key)
          if scene_gid is None:
            continue
          source_gid = g.id
          geom_size[worlds, scene_gid] = v_model.geom_size[source_gid]
          geom_rbound[worlds, scene_gid] = v_model.geom_rbound[source_gid]
          geom_aabb[worlds, scene_gid] = v_model.geom_aabb[source_gid].reshape(2, 3)
          geom_pos[worlds, scene_gid] = v_model.geom_pos[source_gid]
          geom_quat[worlds, scene_gid] = v_model.geom_quat[source_gid]

      # Propagate the variant root's subtreemass delta up the ancestor
      # chain. Base ancestor subtreemass already includes the placeholder
      # contribution from the merged scene compile; adding the delta
      # swaps that placeholder contribution for the variant's actual
      # contribution.
      delta = (
        float(v_model.body_subtreemass[source_body_id_by_path[root_body_path]])
        - base_root_subtreemass
      )
      for anc_bid in ancestor_ids:
        body_subtreemass[worlds, anc_bid] += delta

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


def build_variant_model(
  spec: mujoco.MjSpec,
  nworld: int,
  variant_info: list[tuple[str, VariantMetadata]],
  configure_model: Callable[[mujoco.MjModel], None] | None = None,
) -> MeshVariantResult:
  """Build a Warp Model with per-world mesh assignments.

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
    slots = metadata.slots
    nslots = len(slots)
    nvariants = len(metadata.variant_names)

    # Resolve world-to-variant assignment. The metadata's ``assignment``
    # is either a tuple of weights (from None or dict on the cfg) or a
    # callable. Use ``callable()`` for the dispatch since tuples are
    # not callable.
    assignment_spec = metadata.assignment
    if callable(assignment_spec):
      assignment = list(assignment_spec(nworld))
      if len(assignment) != nworld:
        raise ValueError(
          f"VariantEntityCfg.assignment (entity '{entity_prefix}') "
          f"returned {len(assignment)} indices but nworld={nworld}."
        )
      for w, v in enumerate(assignment):
        if not (0 <= int(v) < nvariants):
          raise ValueError(
            f"VariantEntityCfg.assignment (entity '{entity_prefix}') "
            f"returned variant index {v} at world {w}, but only "
            f"{nvariants} variants are declared."
          )
    else:
      assignment = allocate_worlds(assignment_spec, nworld)
    w2v = np.array(assignment, dtype=np.int32)
    world_to_variant[entity_prefix] = w2v

    # Resolve slot geom IDs by template name (deterministic, no
    # positional or unnamed-padding heuristics).
    slot_geom_ids = np.zeros(nslots, dtype=np.int64)
    for s_idx, slot in enumerate(slots):
      full_geom_name = _qualified_slot_geom_name(entity_prefix, slot.template_geom_name)
      gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, full_geom_name)
      if gid < 0:
        raise ValueError(
          f"Slot geom '{full_geom_name}' (entity '{entity_prefix}', "
          f"slot {slot.key}) not found in compiled model."
        )
      slot_geom_ids[s_idx] = gid

    # Resolve every (variant, slot) -> mesh_id. ``-1`` for slots a
    # variant doesn't fill.
    variant_slot_mesh_ids = np.full((nvariants, nslots), -1, dtype=np.int64)
    for v_idx, slot_specs in enumerate(metadata.variant_slot_specs):
      variant_name = metadata.variant_names[v_idx]
      for s_idx, gspec in enumerate(slot_specs):
        if gspec is None:
          continue
        full_mesh_name = _qualified_mesh_name(
          entity_prefix, variant_name, gspec.mesh_name
        )
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, full_mesh_name)
        if mid < 0:
          raise ValueError(
            f"Mesh '{full_mesh_name}' (variant '{variant_name}', "
            f"slot {slots[s_idx].key}) not found in compiled model."
          )
        variant_slot_mesh_ids[v_idx, s_idx] = mid

    # Vectorized scatter: per-world row from variant assignment.
    dataid_table[:, slot_geom_ids] = variant_slot_mesh_ids[w2v]

  # Build warp model.
  m = mjwarp.put_model(model)
  m.geom_dataid = wp.array(dataid_table, dtype=int)

  # Populate dependent per-world fields.
  _populate_dependent_fields(m, model, nworld, variant_info, world_to_variant)

  return MeshVariantResult(
    wp_model=m,
    mj_model=model,
    world_to_variant=world_to_variant,
  )
