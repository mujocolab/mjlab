"""Tests for per-world mesh variant support."""

from __future__ import annotations

from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import (
  EntityCfg,
  VariantEntityCfg,
)
from mjlab.entity.variants import (
  SlotKey,
  VariantGeomSpec,
  VariantSlot,
  allocate_worlds,
  build_variant_model,
)
from mjlab.viewer.model_sync import (
  disable_model_sameframe_shortcuts,
  sync_model_fields,
)

# Helpers: variant specs with visual + collision mesh geoms.


def _sphere_2col_spec() -> mujoco.MjSpec:
  """Sphere: 1 visual + 2 collision geoms."""
  spec = mujoco.MjSpec()
  mv = spec.add_mesh()
  mv.name = "visual"
  mv.make_sphere(subdivision=3)
  for i in range(2):
    mc = spec.add_mesh()
    mc.name = f"col_{i}"
    mc.make_sphere(subdivision=1)
  body = spec.worldbody.add_body()
  body.name = "prop"
  body.add_freejoint()
  gv = body.add_geom()
  gv.name = "visual"
  gv.type = mujoco.mjtGeom.mjGEOM_MESH
  gv.meshname = "visual"
  gv.contype = 0
  gv.conaffinity = 0
  for i in range(2):
    gc = body.add_geom()
    gc.name = f"col_{i}"
    gc.type = mujoco.mjtGeom.mjGEOM_MESH
    gc.meshname = f"col_{i}"
  return spec


def _cone_4col_spec() -> mujoco.MjSpec:
  """Cone: 1 visual + 4 collision geoms (more than sphere)."""
  spec = mujoco.MjSpec()
  mv = spec.add_mesh()
  mv.name = "visual"
  mv.make_cone(nedge=8, radius=0.05)
  for i in range(4):
    mc = spec.add_mesh()
    mc.name = f"col_{i}"
    mc.make_sphere(subdivision=1)
  body = spec.worldbody.add_body()
  body.name = "prop"
  body.add_freejoint()
  gv = body.add_geom()
  gv.name = "visual"
  gv.type = mujoco.mjtGeom.mjGEOM_MESH
  gv.meshname = "visual"
  gv.contype = 0
  gv.conaffinity = 0
  for i in range(4):
    gc = body.add_geom()
    gc.name = f"col_{i}"
    gc.type = mujoco.mjtGeom.mjGEOM_MESH
    gc.meshname = f"col_{i}"
  return spec


def _simple_sphere_spec() -> mujoco.MjSpec:
  """Single-geom sphere for simple tests."""
  spec = mujoco.MjSpec()
  m = spec.add_mesh()
  m.name = "sphere"
  m.make_sphere(subdivision=2)
  body = spec.worldbody.add_body()
  body.name = "prop"
  body.add_freejoint()
  g = body.add_geom()
  g.name = "visual"
  g.type = mujoco.mjtGeom.mjGEOM_MESH
  g.meshname = "sphere"
  return spec


def _simple_cone_spec() -> mujoco.MjSpec:
  """Single-geom cone for simple tests."""
  spec = mujoco.MjSpec()
  m = spec.add_mesh()
  m.name = "cone"
  m.make_cone(nedge=8, radius=0.05)
  body = spec.worldbody.add_body()
  body.name = "prop"
  body.add_freejoint()
  g = body.add_geom()
  g.name = "visual"
  g.type = mujoco.mjtGeom.mjGEOM_MESH
  g.meshname = "cone"
  return spec


def _hinge_spec() -> mujoco.MjSpec:
  """Object with a hinge joint (incompatible with freejoint variants)."""
  spec = mujoco.MjSpec()
  m = spec.add_mesh()
  m.name = "box"
  m.make_sphere(subdivision=1)
  body = spec.worldbody.add_body()
  body.name = "prop"
  j = body.add_joint()
  j.name = "hinge"
  j.type = mujoco.mjtJoint.mjJNT_HINGE
  g = body.add_geom()
  g.name = "visual"
  g.type = mujoco.mjtGeom.mjGEOM_MESH
  g.meshname = "box"
  return spec


def _build_scene_with_variants(
  variant_a_fn, variant_b_fn, *, weight_a=0.5, weight_b=0.5
):
  """Build a scene spec + variant_info from two variant spec_fns."""
  cfg = VariantEntityCfg(
    variants={"a": variant_a_fn, "b": variant_b_fn},
    assignment={"a": weight_a, "b": weight_b},
  )
  entity = cfg.build()
  assert entity.variant_metadata is not None
  scene_spec = mujoco.MjSpec()
  frame = scene_spec.worldbody.add_frame()
  scene_spec.attach(entity.spec, prefix="object/", frame=frame)
  return scene_spec, [("object/", entity.variant_metadata)]


# allocate_worlds.


def test_allocate_worlds_proportional():
  result = allocate_worlds((0.6, 0.4), 10)
  assert len(result) == 10
  assert result.count(0) == 6
  assert result.count(1) == 4


def test_allocate_worlds_uniform():
  result = allocate_worlds((1.0, 1.0), 8)
  assert result.count(0) == 4
  assert result.count(1) == 4


def test_allocate_worlds_single_variant():
  result = allocate_worlds((1.0,), 5)
  assert result == [0, 0, 0, 0, 0]


def test_allocate_worlds_zero_weight_skips_variant():
  """A zero-weight variant gets zero worlds; the rest split nworld."""
  result = allocate_worlds((1.0, 0.0, 1.0), 10)
  assert len(result) == 10
  assert result.count(1) == 0
  assert result.count(0) == 5
  assert result.count(2) == 5


def test_allocate_worlds_rejects_negative_weight():
  with pytest.raises(ValueError, match="non-negative"):
    allocate_worlds((1.0, -0.1), 10)


def test_allocate_worlds_rejects_all_zero():
  with pytest.raises(ValueError, match="positive sum"):
    allocate_worlds((0.0, 0.0), 10)


def test_allocate_worlds_largest_remainder_sums_to_nworld():
  """Largest-remainder rounding must always allocate exactly nworld worlds."""
  for nworld in (3, 7, 100, 1000):
    result = allocate_worlds((1.0, 1.0, 1.0), nworld)
    assert len(result) == nworld
    # Difference between any two variant counts is at most 1 (uniform).
    counts = [result.count(i) for i in range(3)]
    assert max(counts) - min(counts) <= 1


# assignment_fn override.


def test_assignment_fn_overrides_weights():
  """When assignment_fn is set, it dictates the per-world variant indices."""
  cfg = VariantEntityCfg(
    variants={
      "sphere": _simple_sphere_spec,
      "cone": _simple_cone_spec,  # ignored
    },
    assignment=lambda nworld: [0, 1, 0, 1] * (nworld // 4),
  )
  entity = cfg.build()
  assert entity.variant_metadata is not None
  scene_spec = mujoco.MjSpec()
  frame = scene_spec.worldbody.add_frame()
  scene_spec.attach(entity.spec, prefix="object/", frame=frame)
  result = build_variant_model(scene_spec, 8, [("object/", entity.variant_metadata)])
  w2v = result.world_to_variant["object/"]
  assert list(w2v) == [0, 1, 0, 1, 0, 1, 0, 1]


def test_assignment_fn_seeded_is_nworld_invariant():
  """Per-world independent RNG draws make world W's variant a function of W
  alone, independent of nworld."""
  weights = (1.0, 2.0, 1.0)
  cum = np.cumsum(np.asarray(weights) / sum(weights))

  def seeded_assignment(seed: int):
    def fn(nworld: int) -> list[int]:
      return [
        int(np.searchsorted(cum, np.random.default_rng((seed, w)).random()))
        for w in range(nworld)
      ]

    return fn

  fn_64 = seeded_assignment(seed=42)(64)
  fn_256 = seeded_assignment(seed=42)(256)
  # World 0..63 must agree across both batch sizes.
  assert fn_64 == fn_256[:64]


def test_assignment_fn_rejects_wrong_length():
  cfg = VariantEntityCfg(
    variants={
      "sphere": _simple_sphere_spec,
      "cone": _simple_cone_spec,
    },
    assignment=lambda nworld: [0] * (nworld - 1),  # one short
  )
  entity = cfg.build()
  assert entity.variant_metadata is not None
  scene_spec = mujoco.MjSpec()
  frame = scene_spec.worldbody.add_frame()
  scene_spec.attach(entity.spec, prefix="object/", frame=frame)
  with pytest.raises(ValueError, match="returned .* indices but nworld="):
    build_variant_model(scene_spec, 4, [("object/", entity.variant_metadata)])


def test_assignment_fn_rejects_out_of_range_index():
  cfg = VariantEntityCfg(
    variants={
      "sphere": _simple_sphere_spec,
      "cone": _simple_cone_spec,
    },
    assignment=lambda nworld: [0, 1, 0, 99],  # 99 is out of range
  )
  entity = cfg.build()
  assert entity.variant_metadata is not None
  scene_spec = mujoco.MjSpec()
  frame = scene_spec.worldbody.add_frame()
  scene_spec.attach(entity.spec, prefix="object/", frame=frame)
  with pytest.raises(ValueError, match="returned variant index 99"):
    build_variant_model(scene_spec, 4, [("object/", entity.variant_metadata)])


# Entity merging.


def test_entity_builds_with_variants():
  cfg = VariantEntityCfg(
    variants={
      "sphere": _simple_sphere_spec,
      "cone": _simple_cone_spec,
    },
  )
  entity = cfg.build()
  meta = entity.variant_metadata
  assert meta is not None
  assert meta.variant_names == ("sphere", "cone")
  assert meta.num_mesh_geoms == 1
  mesh_names = [m.name for m in entity.spec.meshes]
  assert any("sphere" in n for n in mesh_names)
  assert any("cone" in n for n in mesh_names)


def test_multi_geom_body_padding():
  """Sphere (3 geoms) + cone (5 geoms) -> body padded to 5 mesh geoms."""
  cfg = VariantEntityCfg(
    variants={
      "sphere": _sphere_2col_spec,
      "cone": _cone_4col_spec,
    },
  )
  entity = cfg.build()
  meta = entity.variant_metadata
  assert meta is not None
  assert meta.num_mesh_geoms == 5  # max(3, 5)
  # Sphere: 3 real + 2 padding (None).
  assert sum(1 for n in meta.variant_mesh_names[0] if n is None) == 2
  # Cone: 5 real, no padding.
  assert all(n is not None for n in meta.variant_mesh_names[1])


# Validation.


def test_mismatched_joint_structure_raises():
  cfg = VariantEntityCfg(
    variants={
      "sphere": _simple_sphere_spec,
      "hinge": _hinge_spec,
    },
  )
  with pytest.raises(ValueError, match="joint"):
    cfg.build()


def test_single_variant_builds():
  """A single variant degenerates cleanly; useful for templated variant sets."""
  cfg = VariantEntityCfg(
    variants={"only": _simple_sphere_spec},
  )
  entity = cfg.build()
  assert entity.variant_metadata is not None
  assert entity.variant_metadata.variant_names == ("only",)


def test_empty_variants_raises():
  cfg = VariantEntityCfg(variants={})
  with pytest.raises(ValueError, match="at least one"):
    cfg.build()


def _fixed_base_sphere_spec() -> mujoco.MjSpec:
  """Fixed-base sphere variant (no free joint): currently unsupported."""
  spec = mujoco.MjSpec()
  m = spec.add_mesh(name="sphere")
  m.make_sphere(subdivision=2)
  body = spec.worldbody.add_body(name="prop")
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="sphere")
  return spec


def test_fixed_base_variants_rejected():
  """Variants must be floating-base; fixed-base raises with a clear message."""
  cfg = VariantEntityCfg(
    variants={
      "a": _fixed_base_sphere_spec,
      "b": _fixed_base_sphere_spec,
    },
  )
  with pytest.raises(ValueError, match="floating-base"):
    cfg.build()


def test_setting_spec_fn_on_variant_cfg_raises():
  """VariantEntityCfg.spec_fn is unused; setting it should fail loudly."""
  with pytest.raises(ValueError, match="spec_fn cannot be set"):
    VariantEntityCfg(
      variants={"only": _simple_sphere_spec},
      spec_fn=_simple_sphere_spec,
    )


# Recursive validation: helpers and tests.


def _articulated_spec(
  *,
  root_mesh: str = "root_mesh",
  child_mesh: str = "child_mesh",
  with_grandchild: bool = False,
) -> mujoco.MjSpec:
  """Root + child body (hinge joint). Optional grandchild for arity tests."""
  spec = mujoco.MjSpec()
  rm = spec.add_mesh(name=root_mesh)
  rm.make_sphere(subdivision=2)
  cm = spec.add_mesh(name=child_mesh)
  cm.make_sphere(subdivision=2)
  root = spec.worldbody.add_body(name="prop")
  root.add_freejoint()
  rg = root.add_geom()
  rg.name = "root_geom"
  rg.type = mujoco.mjtGeom.mjGEOM_MESH
  rg.meshname = root_mesh
  child = root.add_body(name="lid")
  cj = child.add_joint()
  cj.name = "hinge"
  cj.type = mujoco.mjtJoint.mjJNT_HINGE
  cg = child.add_geom()
  cg.name = "child_geom"
  cg.type = mujoco.mjtGeom.mjGEOM_MESH
  cg.meshname = child_mesh
  if with_grandchild:
    gm = spec.add_mesh(name="grand_mesh")
    gm.make_sphere(subdivision=2)
    grand = child.add_body(name="grand")
    grand.add_geom(
      name="grand_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="grand_mesh"
    )
  return spec


def _spec_with_actuator(actuator_name: str = "act") -> mujoco.MjSpec:
  """Single-body sphere with a hinge child + a position actuator."""
  spec = mujoco.MjSpec()
  m = spec.add_mesh(name="sphere")
  m.make_sphere(subdivision=2)
  root = spec.worldbody.add_body(name="prop")
  root.add_freejoint()
  root.add_geom(name="visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="sphere")
  child = root.add_body(name="lid")
  cj = child.add_joint()
  cj.name = "hinge"
  cj.type = mujoco.mjtJoint.mjJNT_HINGE
  child.add_geom(name="lid_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="sphere")
  act = spec.add_actuator()
  act.name = actuator_name
  act.set_to_motor()
  act.target = "hinge"
  return spec


def _spec_with_primitive(primitive_role: str = "collision") -> mujoco.MjSpec:
  """Sphere variant with an additional primitive box (visual or collision)."""
  spec = mujoco.MjSpec()
  m = spec.add_mesh(name="sphere")
  m.make_sphere(subdivision=2)
  body = spec.worldbody.add_body(name="prop")
  body.add_freejoint()
  box = body.add_geom()
  box.name = "primitive"
  box.type = mujoco.mjtGeom.mjGEOM_BOX
  box.size = np.array([0.05, 0.05, 0.05])
  if primitive_role == "visual":
    box.contype = 0
    box.conaffinity = 0
  body.add_geom(name="mesh_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="sphere")
  return spec


def _spec_with_diagonal_inertia() -> mujoco.MjSpec:
  spec = _simple_sphere_spec()
  body = list(spec.worldbody.bodies)[0]
  body.explicitinertial = 1
  body.mass = 1.0
  body.ipos = np.array([0.0, 0.0, 0.0])
  body.inertia = np.array([0.001, 0.001, 0.001])
  body.iquat = np.array([1.0, 0.0, 0.0, 0.0])
  return spec


def _spec_with_fullinertia() -> mujoco.MjSpec:
  spec = _simple_sphere_spec()
  body = list(spec.worldbody.bodies)[0]
  body.explicitinertial = 1
  body.mass = 1.0
  body.ipos = np.array([0.0, 0.0, 0.0])
  body.fullinertia = np.array([0.001, 0.001, 0.001, 0.0, 0.0, 0.0])
  return spec


def _spec_with_reserved_mesh_name() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  m = spec.add_mesh(name="mjlab/pad/sneaky")
  m.make_sphere(subdivision=2)
  body = spec.worldbody.add_body(name="prop")
  body.add_freejoint()
  body.add_geom(
    name="visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mjlab/pad/sneaky"
  )
  return spec


def test_articulated_same_topology_validates():
  """Articulated variants with matching topology pass validation
  (build still rejects via floating-base check; this verifies validation
  itself does not complain)."""
  cfg = VariantEntityCfg(
    variants={
      "a": lambda: _articulated_spec(root_mesh="r_a", child_mesh="c_a"),
      "b": lambda: _articulated_spec(root_mesh="r_b", child_mesh="c_b"),
    },
  )
  entity = cfg.build()
  assert entity.variant_metadata is not None


def test_recursive_child_body_count_mismatch_rejected():
  """Variants with different grandchild counts fail recursive validation."""
  cfg = VariantEntityCfg(
    variants={
      "shallow": lambda: _articulated_spec(with_grandchild=False),
      "deep": lambda: _articulated_spec(with_grandchild=True),
    },
  )
  with pytest.raises(ValueError, match="child bodies"):
    cfg.build()


def test_recursive_child_body_name_mismatch_rejected():
  def variant_lid():
    return _articulated_spec()

  def variant_renamed_child():
    spec = _articulated_spec()
    list(spec.worldbody.bodies)[0].bodies[0].name = "drawer_top"
    return spec

  cfg = VariantEntityCfg(
    variants={
      "lid": variant_lid,
      "drawer": variant_renamed_child,
    },
  )
  with pytest.raises(ValueError, match="body path"):
    cfg.build()


def test_recursive_joint_mismatch_in_child_body_rejected():
  def variant_a():
    return _articulated_spec()

  def variant_b_slide():
    spec = _articulated_spec()
    child = list(spec.worldbody.bodies)[0].bodies[0]
    list(child.joints)[0].type = mujoco.mjtJoint.mjJNT_SLIDE
    return spec

  cfg = VariantEntityCfg(
    variants={
      "hinge": variant_a,
      "slide": variant_b_slide,
    },
  )
  with pytest.raises(ValueError, match="joint"):
    cfg.build()


def test_primitive_geom_count_mismatch_rejected():
  cfg = VariantEntityCfg(
    variants={
      "with_box": _spec_with_primitive,
      "without_box": _simple_sphere_spec,
    },
  )
  with pytest.raises(ValueError, match="non-mesh geoms"):
    cfg.build()


def test_primitive_geom_role_mismatch_rejected():
  cfg = VariantEntityCfg(
    variants={
      "col": lambda: _spec_with_primitive("collision"),
      "vis": lambda: _spec_with_primitive("visual"),
    },
  )
  with pytest.raises(ValueError, match="primitive geom"):
    cfg.build()


def test_actuator_count_mismatch_rejected():
  cfg = VariantEntityCfg(
    variants={
      "no_act": _articulated_spec,
      "with_act": _spec_with_actuator,
    },
  )
  with pytest.raises(ValueError, match="actuator count"):
    cfg.build()


def test_actuator_name_mismatch_rejected():
  cfg = VariantEntityCfg(
    variants={
      "act_a": lambda: _spec_with_actuator("motor_a"),
      "act_b": lambda: _spec_with_actuator("motor_b"),
    },
  )
  with pytest.raises(ValueError, match="actuator #0"):
    cfg.build()


def test_fullinertia_diagonal_mixing_rejected():
  cfg = VariantEntityCfg(
    variants={
      "diag": _spec_with_diagonal_inertia,
      "full": _spec_with_fullinertia,
    },
  )
  with pytest.raises(ValueError, match="inertial representation"):
    cfg.build()


def test_diagonal_inertia_consistent_accepted():
  cfg = VariantEntityCfg(
    variants={
      "a": _spec_with_diagonal_inertia,
      "b": _spec_with_diagonal_inertia,
    },
  )
  entity = cfg.build()
  assert entity.variant_metadata is not None


def test_reserved_prefix_in_mesh_rejected():
  cfg = VariantEntityCfg(
    variants={
      "good": _simple_sphere_spec,
      "bad": _spec_with_reserved_mesh_name,
    },
  )
  with pytest.raises(ValueError, match="reserved name prefix"):
    cfg.build()


def test_validation_error_format():
  """Error messages use the standardized mjlab.entity prefix and Hint suffix."""
  cfg = VariantEntityCfg(
    variants={
      "ok": _simple_sphere_spec,
      "bad": _hinge_spec,
    },
  )
  with pytest.raises(ValueError) as excinfo:
    cfg.build()
  msg = str(excinfo.value)
  assert msg.startswith("mjlab.entity: VariantEntityCfg 'bad': ")
  assert "Hint:" in msg


def _spec_with_sensor() -> mujoco.MjSpec:
  """Sphere with a free joint and a velocity sensor on it."""
  spec = _simple_sphere_spec()
  s = spec.add_sensor()
  s.name = "vel"
  s.type = mujoco.mjtSensor.mjSENS_VELOCIMETER
  s.objtype = mujoco.mjtObj.mjOBJ_SITE
  s.objname = "site_a"
  # Sensor needs a site target; add one.
  body = list(spec.worldbody.bodies)[0]
  site = body.add_site()
  site.name = "site_a"
  return spec


def test_sensor_count_mismatch_rejected():
  cfg = VariantEntityCfg(
    variants={
      "no_sens": _simple_sphere_spec,
      "with_sens": _spec_with_sensor,
    },
  )
  with pytest.raises(ValueError, match="sensor count"):
    cfg.build()


def test_validate_specs_directly_rejects_zero_root_bodies():
  """Empty worldbody fails with a clear root-body message."""
  from mjlab.entity.variants import validate_variant_specs

  empty = mujoco.MjSpec()
  ok = _simple_sphere_spec()
  with pytest.raises(ValueError, match="exactly one root body"):
    validate_variant_specs(["empty", "ok"], [empty, ok])


# Slot metadata (Workstream 3).


def _slot_is_padding(meta, slot_index: int) -> bool:
  """True if any variant leaves the slot at ``slot_index`` unfilled."""
  return any(specs[slot_index] is None for specs in meta.variant_slot_specs)


def test_slot_metadata_single_variant_single_geom():
  cfg = VariantEntityCfg(variants={"only": _simple_sphere_spec})
  meta = cfg.build().variant_metadata
  assert meta is not None
  assert len(meta.slots) == 1
  slot = meta.slots[0]
  assert isinstance(slot, VariantSlot)
  assert slot.key == SlotKey(body_path="/prop", role="collision", ordinal=0)
  assert _slot_is_padding(meta, 0) is False
  assert slot.template_geom_name == "mjlab/pad/prop/collision/0"
  # source_geom_names has one entry per variant.
  assert slot.source_geom_names == ("visual",)
  # variant_slot_specs aligns with slots.
  assert len(meta.variant_slot_specs) == 1
  assert len(meta.variant_slot_specs[0]) == 1
  vgs = meta.variant_slot_specs[0][0]
  assert isinstance(vgs, VariantGeomSpec)
  assert vgs.geom_name == "visual"
  assert vgs.mesh_name == "sphere"


def test_slot_metadata_visual_collision_split():
  """Visual and collision geoms on the same body get distinct slots."""
  cfg = VariantEntityCfg(
    variants={
      "sphere": _sphere_2col_spec,
      "cone": _cone_4col_spec,
    }
  )
  meta = cfg.build().variant_metadata
  assert meta is not None
  # Body /prop has 1 visual + max(2, 4) = 4 collision slots.
  visual_slots = [s for s in meta.slots if s.key.role == "visual"]
  collision_slots = [s for s in meta.slots if s.key.role == "collision"]
  assert len(visual_slots) == 1
  assert len(collision_slots) == 4
  assert all(s.key.body_path == "/prop" for s in meta.slots)


def test_slot_metadata_visual_before_collision():
  """Slot ordering: visual slots come before collision slots per body."""
  cfg = VariantEntityCfg(
    variants={
      "sphere": _sphere_2col_spec,
      "cone": _cone_4col_spec,
    }
  )
  meta = cfg.build().variant_metadata
  assert meta is not None
  roles = [s.key.role for s in meta.slots]
  # Find first collision; all visual should come before it.
  first_col = roles.index("collision")
  assert all(r == "visual" for r in roles[:first_col])
  assert all(r == "collision" for r in roles[first_col:])


def test_slot_metadata_padding_derivable_from_specs():
  """A slot held by some variants but not others is unfilled (None) for those."""
  cfg = VariantEntityCfg(
    variants={
      "sphere": _sphere_2col_spec,
      "cone": _cone_4col_spec,
    }
  )
  meta = cfg.build().variant_metadata
  assert meta is not None
  col_positions = [i for i, s in enumerate(meta.slots) if s.key.role == "collision"]
  # Sphere has 2 collision; cone has 4. Slots 0, 1 fully populated; 2, 3 are padding.
  assert _slot_is_padding(meta, col_positions[0]) is False
  assert _slot_is_padding(meta, col_positions[1]) is False
  assert _slot_is_padding(meta, col_positions[2]) is True
  assert _slot_is_padding(meta, col_positions[3]) is True
  # Sphere has None at the padding slots.
  variant_idx_sphere = meta.variant_names.index("sphere")
  specs_sphere = meta.variant_slot_specs[variant_idx_sphere]
  assert specs_sphere[col_positions[2]] is None
  assert specs_sphere[col_positions[3]] is None
  # Cone has VariantGeomSpec for all collision slots.
  variant_idx_cone = meta.variant_names.index("cone")
  specs_cone = meta.variant_slot_specs[variant_idx_cone]
  for cp in col_positions:
    assert specs_cone[cp] is not None


def test_slot_metadata_articulated_per_body_slots():
  """Articulated variants produce slots per (body, role)."""
  cfg = VariantEntityCfg(
    variants={
      "a": lambda: _articulated_spec(root_mesh="ar", child_mesh="ac"),
      "b": lambda: _articulated_spec(root_mesh="br", child_mesh="bc"),
    }
  )
  meta = cfg.build().variant_metadata
  assert meta is not None
  # Two bodies (/prop, /prop/lid), each with 1 collision mesh -> 2 slots.
  paths = sorted({s.key.body_path for s in meta.slots})
  assert paths == ["/prop", "/prop/lid"]
  # Each body has exactly one collision slot, no visuals.
  for path in paths:
    body_slots = [s for s in meta.slots if s.key.body_path == path]
    assert len(body_slots) == 1
    assert body_slots[0].key.role == "collision"
    slot_idx = meta.slots.index(body_slots[0])
    assert _slot_is_padding(meta, slot_idx) is False
  # Mesh names captured per variant per slot.
  variant_a_specs = meta.variant_slot_specs[meta.variant_names.index("a")]
  variant_b_specs = meta.variant_slot_specs[meta.variant_names.index("b")]
  a_meshes = sorted(s.mesh_name for s in variant_a_specs if s is not None)
  b_meshes = sorted(s.mesh_name for s in variant_b_specs if s is not None)
  assert a_meshes == ["ac", "ar"]
  assert b_meshes == ["bc", "br"]


def test_slot_metadata_template_name_under_reserved_prefix():
  """Every template slot name starts with the reserved mjlab/pad/ prefix."""
  cfg = VariantEntityCfg(
    variants={
      "a": _simple_sphere_spec,
      "b": _simple_cone_spec,
    }
  )
  meta = cfg.build().variant_metadata
  assert meta is not None
  for slot in meta.slots:
    assert slot.template_geom_name.startswith("mjlab/pad/")


def test_slot_metadata_captures_visual_role_from_zero_contact_bits():
  """A geom with contype=0 and conaffinity=0 is classified as visual."""
  cfg = VariantEntityCfg(
    variants={
      "sphere": _sphere_2col_spec,
      "cone": _cone_4col_spec,
    }
  )
  meta = cfg.build().variant_metadata
  assert meta is not None
  visual_slots = [s for s in meta.slots if s.key.role == "visual"]
  assert len(visual_slots) == 1
  visual_slot = visual_slots[0]
  for variant_specs in meta.variant_slot_specs:
    spec_at_visual = variant_specs[meta.slots.index(visual_slot)]
    assert spec_at_visual is not None
    assert spec_at_visual.contype == 0
    assert spec_at_visual.conaffinity == 0


def test_slot_metadata_ordinals_are_zero_based_per_body_and_role():
  """Within a (body, role), slot ordinals are 0..max-1."""
  cfg = VariantEntityCfg(
    variants={
      "sphere": _sphere_2col_spec,
      "cone": _cone_4col_spec,
    }
  )
  meta = cfg.build().variant_metadata
  assert meta is not None
  collision_slots = [s for s in meta.slots if s.key.role == "collision"]
  assert [s.key.ordinal for s in collision_slots] == [0, 1, 2, 3]
  visual_slots = [s for s in meta.slots if s.key.role == "visual"]
  assert [s.key.ordinal for s in visual_slots] == [0]


def test_slot_metadata_alignment_invariant():
  """variant_slot_specs[v] aligns with slots positionally for every variant."""
  cfg = VariantEntityCfg(
    variants={
      "sphere": _sphere_2col_spec,
      "cone": _cone_4col_spec,
    }
  )
  meta = cfg.build().variant_metadata
  assert meta is not None
  for variant_specs in meta.variant_slot_specs:
    assert len(variant_specs) == len(meta.slots)


def test_template_geom_contype_matches_slot_role():
  """Template contype/conaffinity are derived from slot role (union of variants)."""
  scene_spec, vi = _build_scene_with_variants(_sphere_2col_spec, _cone_4col_spec)
  result = build_variant_model(scene_spec, 4, vi)
  metadata = vi[0][1]
  for slot in metadata.slots:
    full_name = f"object/{slot.template_geom_name}"
    gid = mujoco.mj_name2id(result.mj_model, mujoco.mjtObj.mjOBJ_GEOM, full_name)
    assert gid >= 0, f"slot geom '{full_name}' missing from compiled model"
    contype = int(result.mj_model.geom_contype[gid])
    conaffinity = int(result.mj_model.geom_conaffinity[gid])
    if slot.key.role == "visual":
      assert contype == 0, f"visual slot {slot.key} has contype={contype}"
      assert conaffinity == 0, f"visual slot {slot.key} has conaffinity={conaffinity}"
    else:
      assert contype == 1, f"collision slot {slot.key} has contype={contype}"
      assert conaffinity == 1, (
        f"collision slot {slot.key} has conaffinity={conaffinity}"
      )


def test_template_geoms_use_mjlab_pad_prefix():
  """All entity mesh-geom names in the compiled template use mjlab/pad/ prefix."""
  scene_spec, vi = _build_scene_with_variants(_sphere_2col_spec, _cone_4col_spec)
  result = build_variant_model(scene_spec, 4, vi)
  for gid in range(result.mj_model.ngeom):
    if result.mj_model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    name = mujoco.mj_id2name(result.mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    if not name.startswith("object/"):
      continue
    suffix = name[len("object/") :]
    assert suffix.startswith("mjlab/pad/"), (
      f"entity mesh geom '{name}' does not use mjlab/pad/ prefix"
    )


def test_visual_collision_split_inertia_matches_independent_compile():
  """Per-world body_mass matches independent compile when variants have visual+collision split.

  This verifies the slot-driven reference compile preserves the visual
  role (contype=0/conaffinity=0) on the visual mesh; if the old
  contype=1/conaffinity=1 reset still ran, the visual mesh's
  inertia-inference behavior would not change for default groups, but
  this exercise pins the contract end-to-end.
  """
  scene_spec, vi = _build_scene_with_variants(_sphere_2col_spec, _cone_4col_spec)
  result = build_variant_model(scene_spec, 4, vi)

  sphere_model = _sphere_2col_spec().compile()
  cone_model = _cone_4col_spec().compile()

  body_mass = result.wp_model.body_mass.numpy()
  w2v = result.world_to_variant["object/"]
  obj_body = result.mj_model.nbody - 1

  sphere_w = int(np.where(w2v == 0)[0][0])
  cone_w = int(np.where(w2v == 1)[0][0])

  np.testing.assert_allclose(
    body_mass[sphere_w, obj_body],
    sphere_model.body_mass[-1],
    atol=1e-4,
  )
  np.testing.assert_allclose(
    body_mass[cone_w, obj_body],
    cone_model.body_mass[-1],
    atol=1e-4,
  )


def test_variant_order_irrelevant_per_variant_compile():
  """Reordering the variant dict does not change per-variant per-world fields."""
  cfg_ab = VariantEntityCfg(
    variants={
      "a": _simple_sphere_spec,
      "b": _simple_cone_spec,
    }
  )
  cfg_ba = VariantEntityCfg(
    variants={
      "b": _simple_cone_spec,
      "a": _simple_sphere_spec,
    }
  )

  def _build_scene(cfg: VariantEntityCfg):
    entity = cfg.build()
    assert entity.variant_metadata is not None
    scene_spec = mujoco.MjSpec()
    frame = scene_spec.worldbody.add_frame()
    scene_spec.attach(entity.spec, prefix="object/", frame=frame)
    return scene_spec, [("object/", entity.variant_metadata)]

  scene_ab, vi_ab = _build_scene(cfg_ab)
  scene_ba, vi_ba = _build_scene(cfg_ba)
  res_ab = build_variant_model(scene_ab, 4, vi_ab)
  res_ba = build_variant_model(scene_ba, 4, vi_ba)

  obj_body_ab = res_ab.mj_model.nbody - 1
  obj_body_ba = res_ba.mj_model.nbody - 1

  body_mass_ab = res_ab.wp_model.body_mass.numpy()
  body_mass_ba = res_ba.wp_model.body_mass.numpy()

  # In cfg_ab, "a" is variant index 0; in cfg_ba, "a" is variant index 1.
  w2v_ab = res_ab.world_to_variant["object/"]
  w2v_ba = res_ba.world_to_variant["object/"]
  a_world_ab = int(np.where(w2v_ab == 0)[0][0])
  a_world_ba = int(np.where(w2v_ba == 1)[0][0])
  b_world_ab = int(np.where(w2v_ab == 1)[0][0])
  b_world_ba = int(np.where(w2v_ba == 0)[0][0])

  np.testing.assert_allclose(
    body_mass_ab[a_world_ab, obj_body_ab],
    body_mass_ba[a_world_ba, obj_body_ba],
    atol=1e-5,
    err_msg="variant 'a' body_mass differs across orderings",
  )
  np.testing.assert_allclose(
    body_mass_ab[b_world_ab, obj_body_ab],
    body_mass_ba[b_world_ba, obj_body_ba],
    atol=1e-5,
    err_msg="variant 'b' body_mass differs across orderings",
  )


def test_slot_metadata_source_geom_names_record_padding_as_none():
  """source_geom_names has None where a variant doesn't fill a slot."""
  cfg = VariantEntityCfg(
    variants={
      "sphere": _sphere_2col_spec,
      "cone": _cone_4col_spec,
    }
  )
  meta = cfg.build().variant_metadata
  assert meta is not None
  sphere_idx = meta.variant_names.index("sphere")
  collision_slots = [s for s in meta.slots if s.key.role == "collision"]
  # Sphere has 2 collision -> ordinals 2, 3 are None for sphere.
  assert collision_slots[2].source_geom_names[sphere_idx] is None
  assert collision_slots[3].source_geom_names[sphere_idx] is None
  # Cone fills all 4.
  cone_idx = meta.variant_names.index("cone")
  for cs in collision_slots:
    assert cs.source_geom_names[cone_idx] is not None


def test_no_variants_unchanged():
  cfg = EntityCfg(spec_fn=_simple_sphere_spec)
  entity = cfg.build()
  assert entity.variant_metadata is None


# build_variant_model: dataid and dependent fields.


def test_dataid_assigned_per_world():
  """Each world's geom_dataid points to its variant's meshes."""
  scene_spec, vi = _build_scene_with_variants(_simple_sphere_spec, _simple_cone_spec)
  result = build_variant_model(scene_spec, 4, vi)

  dataid = result.wp_model.geom_dataid.numpy()
  assert dataid.shape == (4, result.mj_model.ngeom)
  assert dataid.ndim == 2

  w2v = result.world_to_variant["object/"]
  assert w2v[0] == 0  # variant a (sphere)
  assert w2v[2] == 1  # variant b (cone)

  # Sphere and cone worlds must have different dataid values.
  assert not np.array_equal(dataid[0], dataid[2])


def test_padding_slots_get_disabled():
  """Shorter variant's padding geom slots have dataid == -1."""
  scene_spec, vi = _build_scene_with_variants(_sphere_2col_spec, _cone_4col_spec)
  result = build_variant_model(scene_spec, 4, vi)

  dataid = result.wp_model.geom_dataid.numpy()
  w2v = result.world_to_variant["object/"]

  # Find a sphere world (variant 0, 3 mesh geoms -> 2 padding slots).
  sphere_world = int(np.where(w2v == 0)[0][0])
  # Find mesh geom columns (skip non-mesh geoms like worldbody).
  mesh_geom_ids = [
    gid
    for gid in range(result.mj_model.ngeom)
    if result.mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
  ]
  sphere_dataid = dataid[sphere_world, mesh_geom_ids]
  # Last 2 mesh geom slots should be -1 (disabled padding).
  assert sphere_dataid[-1] == -1
  assert sphere_dataid[-2] == -1
  # Padding slots must still be collision-enabled in the template/warp model.
  # Short variants are disabled by per-world dataid=-1; long variants need the
  # same slots enabled so their extra hulls can collide.
  assert np.all(result.mj_model.geom_contype[mesh_geom_ids[-2:]] == 1)
  assert np.all(result.mj_model.geom_conaffinity[mesh_geom_ids[-2:]] == 1)
  assert np.all(result.wp_model.geom_contype.numpy()[mesh_geom_ids[-2:]] == 1)
  assert np.all(result.wp_model.geom_conaffinity.numpy()[mesh_geom_ids[-2:]] == 1)
  # First 3 should be valid (>= 0).
  assert all(d >= 0 for d in sphere_dataid[:3])


def test_dependent_fields_match_individual_compilation():
  """Per-world body_mass matches independently compiled variant models."""
  scene_spec, vi = _build_scene_with_variants(_simple_sphere_spec, _simple_cone_spec)
  result = build_variant_model(scene_spec, 4, vi)

  # Compile each variant independently for reference values.
  sphere_model = _simple_sphere_spec().compile()
  cone_model = _simple_cone_spec().compile()

  body_mass = result.wp_model.body_mass.numpy()
  w2v = result.world_to_variant["object/"]

  sphere_w = int(np.where(w2v == 0)[0][0])
  cone_w = int(np.where(w2v == 1)[0][0])

  # The object body is the last body in the scene.
  obj_body = result.mj_model.nbody - 1

  # Mass should match individually compiled models.
  np.testing.assert_allclose(
    body_mass[sphere_w, obj_body],
    sphere_model.body_mass[-1],
    atol=1e-4,
  )
  np.testing.assert_allclose(
    body_mass[cone_w, obj_body],
    cone_model.body_mass[-1],
    atol=1e-4,
  )

  # Sphere and cone should have different masses.
  assert not np.isclose(body_mass[sphere_w, obj_body], body_mass[cone_w, obj_body])


def test_select_default_values_uses_per_world_variant_defaults():
  """Per-world defaults are indexed by env first, then by entity."""
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.envs.mdp.dr._core import _select_default_values
  from mjlab.scene import SceneCfg
  from mjlab.terrains import TerrainEntityCfg

  def _explicit_variant(
    mesh_name: str,
    mass: float,
    inertia: tuple[float, float, float],
    *,
    cone: bool = False,
  ) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    mesh = spec.add_mesh()
    mesh.name = mesh_name
    if cone:
      mesh.make_cone(nedge=8, radius=0.05)
    else:
      mesh.make_sphere(subdivision=1)
    body = spec.worldbody.add_body(name="prop")
    body.add_freejoint()
    body.explicitinertial = 1
    body.mass = mass
    body.ipos[:] = (0.0, 0.0, 0.0)
    body.inertia[:] = inertia
    body.iquat[:] = (1.0, 0.0, 0.0, 0.0)
    body.add_geom(
      name="visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
      contype=0,
      conaffinity=0,
      mass=0.0,
    )
    return spec

  object_cfg = VariantEntityCfg(
    variants={
      "sphere": lambda: _explicit_variant("sphere", 0.2, (1e-4, 2e-4, 3e-4)),
      "cone": lambda: _explicit_variant("cone", 0.7, (4e-4, 5e-4, 6e-4), cone=True),
    },
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
  )
  env_cfg = ManagerBasedRlEnvCfg(
    decimation=1,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4,
      env_spacing=1.0,
      entities={"object": object_cfg},
    ),
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    obj_body = int(env.scene["object"].indexing.root_body_id)
    env_ids = torch.arange(env.num_envs, device=env.device)
    body_ids = torch.tensor([obj_body], device=env.device)

    for field in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
      selected = _select_default_values(env, field, env_ids, body_ids)
      torch.testing.assert_close(
        selected[:, 0],
        getattr(env.sim.model, field)[:, obj_body],
      )
  finally:
    env.close()


def test_viser_builds_per_world_mesh_handles_for_variants():
  """Viser dynamic meshes must not collapse all worlds onto env0's mesh."""
  from contextlib import nullcontext

  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.scene import SceneCfg
  from mjlab.terrains import TerrainEntityCfg
  from mjlab.viewer.viser.scene import MjlabViserScene, _PerWorldMeshGroup

  class _Handle:
    def __init__(self, **kwargs):
      self.visible = kwargs.get("visible", True)
      self.batched_positions = kwargs.get("batched_positions", np.zeros((0, 3)))
      self.batched_wxyzs = kwargs.get("batched_wxyzs", np.zeros((0, 4)))
      self.batched_scales = kwargs.get("batched_scales")
      self.batched_colors = kwargs.get("batched_colors")
      self.batched_opacities = kwargs.get("batched_opacities")
      self.position = kwargs.get("position", np.zeros(3))
      self.wxyz = kwargs.get("wxyz", np.array([1.0, 0.0, 0.0, 0.0]))

    def remove(self) -> None:
      pass

  class _Scene:
    def __init__(self):
      self.batched: list[tuple[tuple, dict, _Handle]] = []

    def configure_environment_map(self, **_kwargs) -> None:
      pass

    def add_frame(self, *_args, **kwargs) -> _Handle:
      return _Handle(**kwargs)

    def add_grid(self, *_args, **kwargs) -> _Handle:
      return _Handle(**kwargs)

    def add_mesh_trimesh(self, *_args, **kwargs) -> _Handle:
      return _Handle(**kwargs)

    def add_batched_meshes_trimesh(self, *args, **kwargs) -> _Handle:
      handle = _Handle(**kwargs)
      self.batched.append((args, kwargs, handle))
      return handle

    def add_batched_meshes_simple(self, *args, **kwargs) -> _Handle:
      handle = _Handle(**kwargs)
      self.batched.append((args, kwargs, handle))
      return handle

  class _Server:
    def __init__(self):
      self.scene = _Scene()

    def atomic(self):
      return nullcontext()

    def flush(self) -> None:
      pass

  env_cfg = ManagerBasedRlEnvCfg(
    decimation=1,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4,
      env_spacing=1.0,
      entities={
        "object": VariantEntityCfg(
          variants={
            "sphere": _simple_sphere_spec,
            "cone": _simple_cone_spec,
          },
          init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )
      },
    ),
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    env.sim.expand_model_fields(("geom_rgba",))
    env.sim.model.geom_rgba[:, :, :3] = torch.linspace(
      0.2,
      0.9,
      env.num_envs,
      device=env.device,
    )[:, None, None]
    server = _Server()
    scene = MjlabViserScene(
      cast(Any, server),
      env.sim.mj_model,
      env.num_envs,
      sim_model=env.sim.model,
      expanded_fields=env.sim.expanded_fields,
    )
    groups = [mg for mg in scene._mesh_groups if isinstance(mg, _PerWorldMeshGroup)]

    assert groups
    assert sum(len(mg.env_ids) for mg in groups) >= env.num_envs

    body_xpos = env.sim.data.xpos.cpu().numpy()
    body_xmat = env.sim.data.xmat.cpu().numpy()
    mocap_pos = (
      env.sim.data.mocap_pos.cpu().numpy() if env.sim.mj_model.nmocap > 0 else None
    )
    mocap_quat = (
      env.sim.data.mocap_quat.cpu().numpy() if env.sim.mj_model.nmocap > 0 else None
    )
    scene.show_only_selected = True
    scene.update_from_arrays(body_xpos, body_xmat, mocap_pos, mocap_quat, env_idx=0)
    scene.update_from_arrays(body_xpos, body_xmat, mocap_pos, mocap_quat, env_idx=1)

    assert any(mg.handle.visible for mg in groups)

    handle_count = len(server.scene.batched)
    env.sim.model.geom_rgba[:, :, :3] = torch.linspace(
      0.9,
      0.2,
      env.num_envs,
      device=env.device,
    )[:, None, None]
    scene.update_from_arrays(body_xpos, body_xmat, mocap_pos, mocap_quat, env_idx=0)
    assert len(server.scene.batched) > handle_count
  finally:
    env.close()


def test_viser_convex_hulls_are_per_variant():
  """Convex-hull handles must differ across variants, not all show env0's hull."""
  from contextlib import nullcontext

  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.scene import SceneCfg
  from mjlab.terrains import TerrainEntityCfg
  from mjlab.viewer.viser.scene import MjlabViserScene, _PerWorldHullGroup

  class _Handle:
    def __init__(self, **kwargs):
      self.visible = kwargs.get("visible", True)
      self.batched_positions = kwargs.get("batched_positions", np.zeros((0, 3)))
      self.batched_wxyzs = kwargs.get("batched_wxyzs", np.zeros((0, 4)))
      self.batched_scales = kwargs.get("batched_scales")
      self.batched_colors = kwargs.get("batched_colors")
      self.batched_opacities = kwargs.get("batched_opacities")
      self.position = kwargs.get("position", np.zeros(3))
      self.wxyz = kwargs.get("wxyz", np.array([1.0, 0.0, 0.0, 0.0]))
      self.vertices = kwargs.get("vertices")
      self.faces = kwargs.get("faces")

    def remove(self) -> None:
      pass

  class _Scene:
    def __init__(self):
      self.batched: list[tuple[tuple, dict, _Handle]] = []

    def configure_environment_map(self, **_kwargs) -> None:
      pass

    def add_frame(self, *_args, **kwargs) -> _Handle:
      return _Handle(**kwargs)

    def add_grid(self, *_args, **kwargs) -> _Handle:
      return _Handle(**kwargs)

    def add_mesh_trimesh(self, *_args, **kwargs) -> _Handle:
      return _Handle(**kwargs)

    def add_batched_meshes_trimesh(self, *args, **kwargs) -> _Handle:
      handle = _Handle(**kwargs)
      self.batched.append((args, kwargs, handle))
      return handle

    def add_batched_meshes_simple(self, path, vertices, faces, **kwargs) -> _Handle:
      # Capture the mesh identity so the test can compare hull shapes.
      kwargs = dict(kwargs)
      kwargs["vertices"] = np.asarray(vertices)
      kwargs["faces"] = np.asarray(faces)
      handle = _Handle(**kwargs)
      self.batched.append(((path,), kwargs, handle))
      return handle

  class _Server:
    def __init__(self):
      self.scene = _Scene()

    def atomic(self):
      return nullcontext()

    def flush(self) -> None:
      pass

  # Sphere and cone produce visibly different convex hulls.
  env_cfg = ManagerBasedRlEnvCfg(
    decimation=1,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4,
      env_spacing=1.0,
      entities={
        "object": VariantEntityCfg(
          variants={
            "sphere": _simple_sphere_spec,
            "cone": _simple_cone_spec,
          },
          init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )
      },
    ),
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    server = _Server()
    scene = MjlabViserScene(
      cast(Any, server),
      env.sim.mj_model,
      env.num_envs,
      sim_model=env.sim.model,
      expanded_fields=env.sim.expanded_fields,
    )
    groups: list[_PerWorldHullGroup] = list(scene._hull_per_world_groups)
    # Two distinct variants -> at least two hull handles on the same body.
    assert len(groups) >= 2, f"expected >=2 hull variants, got {len(groups)}"
    all_envs = np.concatenate([g.env_ids for g in groups])
    assert sorted(all_envs.tolist()) == list(range(env.num_envs))
    # Hulls must be shape-distinct, not all copies of env0's hull.
    shapes = {(g.handle.vertices.shape, g.handle.faces.shape) for g in groups}
    assert len(shapes) >= 2, (
      f"hull variants collapsed to one shape: {shapes} "
      "(all envs would share env0's hull)"
    )

    body_xpos = env.sim.data.xpos.cpu().numpy()
    body_xmat = env.sim.data.xmat.cpu().numpy()
    scene.show_convex_hull = True
    scene.show_only_selected = True
    for target_env in range(env.num_envs):
      scene.update_from_arrays(body_xpos, body_xmat, env_idx=target_env)
      visible_groups = [g for g in groups if g.handle.visible]
      assert len(visible_groups) == 1
      assert target_env in visible_groups[0].env_ids
      assert visible_groups[0].handle.batched_positions.shape[0] == 1

    scene.show_only_selected = False
    scene.update_from_arrays(body_xpos, body_xmat, env_idx=0)
    assert all(g.handle.visible for g in groups)
  finally:
    env.close()


# DR consistency on variant scenes.


def _explicit_mass_variant(
  mesh_name: str,
  mass: float,
  *,
  cone: bool = False,
) -> mujoco.MjSpec:
  """Build a single-geom freejoint variant with an explicit body mass."""
  spec = mujoco.MjSpec()
  mesh = spec.add_mesh()
  mesh.name = mesh_name
  if cone:
    mesh.make_cone(nedge=8, radius=0.05)
  else:
    mesh.make_sphere(subdivision=1)
  body = spec.worldbody.add_body(name="prop")
  body.add_freejoint()
  body.explicitinertial = 1
  body.mass = mass
  body.ipos[:] = (0.0, 0.0, 0.0)
  body.inertia[:] = (1e-4, 1e-4, 1e-4)
  body.iquat[:] = (1.0, 0.0, 0.0, 0.0)
  body.add_geom(
    name="visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=mesh_name,
    contype=0,
    conaffinity=0,
    mass=0.0,
  )
  return spec


def test_dr_body_mass_scale_preserves_variant_baseline():
  """``dr.body_mass`` scale must use each variant's own baseline.

  This is the load-bearing claim of ``_per_world_default_fields``: scaling
  body_mass on a variant scene by a per-env factor must produce
  ``variant_default[env] * scale[env]``, not ``template_default * scale[env]``.
  """
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.envs.mdp import dr
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.scene import SceneCfg
  from mjlab.terrains import TerrainEntityCfg

  light_mass = 0.1
  heavy_mass = 1.0
  scale = 2.0

  object_cfg = VariantEntityCfg(
    variants={
      "light": lambda: _explicit_mass_variant("light", light_mass),
      "heavy": lambda: _explicit_mass_variant("heavy", heavy_mass, cone=True),
    },
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
  )
  env_cfg = ManagerBasedRlEnvCfg(
    decimation=1,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4,
      env_spacing=1.0,
      entities={"object": object_cfg},
    ),
    events={
      "scale_mass": EventTermCfg(
        func=dr.body_mass,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg("object", body_names=("prop",)),
          "operation": "scale",
          "ranges": (scale, scale),  # deterministic factor
        },
      ),
    },
  )

  with pytest.warns(UserWarning, match="dr.body_mass only randomizes mass"):
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    obj_body = int(env.scene["object"].indexing.root_body_id)
    w2v = env.sim.world_to_variant["object"]
    actual = env.sim.model.body_mass[:, obj_body].cpu()

    variant_baseline = torch.tensor([light_mass, heavy_mass], dtype=actual.dtype)
    expected = variant_baseline[w2v.cpu()] * scale
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    # Sanity: at least one env per variant, otherwise the test is vacuous.
    assert (w2v == 0).any() and (w2v == 1).any()
  finally:
    env.close()


# Full env lifecycle.


def test_env_step_with_variants():
  """Build a full ManagerBasedRlEnv with variants; step without crashing."""
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.envs.mdp.events import reset_root_state_uniform
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.scene import SceneCfg
  from mjlab.terrains import TerrainEntityCfg

  object_cfg = VariantEntityCfg(
    variants={
      "sphere": _simple_sphere_spec,
      "cone": _simple_cone_spec,
    },
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    decimation=2,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4,
      env_spacing=1.0,
      entities={"object": object_cfg},
    ),
    events={
      "reset": EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
          "pose_range": {},
          "velocity_range": {},
          "asset_cfg": SceneEntityCfg("object"),
        },
      ),
    },
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  obs, _ = env.reset()
  actions = torch.zeros(env.num_envs, 0)
  for _ in range(10):
    obs, rew, term, trunc, info = env.step(actions)
  # No NaN in positions.
  qpos = env.sim.data.qpos[:].cpu().numpy()
  assert np.all(np.isfinite(qpos))
  env.close()


# Viewer: sameframe shortcut fix.


def _viewer_regression_sphere_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  m = spec.add_mesh()
  m.name = "sphere"
  m.make_sphere(subdivision=3)
  m.scale[:] = (0.05, 0.05, 0.05)
  body = spec.worldbody.add_body()
  body.name = "prop"
  body.add_freejoint()
  g = body.add_geom()
  g.name = "visual"
  g.type = mujoco.mjtGeom.mjGEOM_MESH
  g.meshname = "sphere"
  return spec


def _viewer_regression_cone_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  m = spec.add_mesh()
  m.name = "cone"
  m.make_cone(nedge=16, radius=0.04)
  m.scale[:] = (0.05, 0.05, 0.05)
  body = spec.worldbody.add_body()
  body.name = "prop"
  body.add_freejoint()
  g = body.add_geom()
  g.name = "visual"
  g.type = mujoco.mjtGeom.mjGEOM_MESH
  g.meshname = "cone"
  return spec


def test_sameframe_fix_makes_host_forward_match_variant():
  """Clearing sameframe shortcuts aligns host mj_forward with variant."""
  base_model = _viewer_regression_sphere_spec().compile()
  cone_model = _viewer_regression_cone_spec().compile()

  # Sync cone's kinematic fields onto sphere's model (like viewer does).
  for field in (
    "geom_size",
    "geom_pos",
    "geom_quat",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
  ):
    getattr(base_model, field)[:] = getattr(cone_model, field)

  base_data = mujoco.MjData(base_model)
  base_data.qpos[:] = cone_model.qpos0
  base_data.qpos[2] = 0.05
  mujoco.mj_forward(base_model, base_data)

  cone_data = mujoco.MjData(cone_model)
  cone_data.qpos[:] = cone_model.qpos0
  cone_data.qpos[2] = 0.05
  mujoco.mj_forward(cone_model, cone_data)

  # Before fix: positions differ due to stale sameframe flags.
  assert not np.allclose(base_data.geom_xpos, cone_data.geom_xpos)

  # After fix: clearing sameframe makes them match.
  disable_model_sameframe_shortcuts(base_model)
  mujoco.mj_forward(base_model, base_data)
  np.testing.assert_allclose(base_data.geom_xpos, cone_data.geom_xpos, atol=1e-6)


def test_sync_model_fields_copies_only_requested_env_fields():
  """Viewer model sync copies explicit fields and leaves others unchanged."""
  model = _simple_sphere_spec().compile()

  class _SimModel:
    geom_rgba = torch.tensor(
      [
        [[0.1, 0.2, 0.3, 0.4]],
        [[0.5, 0.6, 0.7, 0.8]],
      ],
      dtype=torch.float32,
    )
    geom_pos = torch.tensor(
      [
        [[1.0, 2.0, 3.0]],
        [[4.0, 5.0, 6.0]],
      ],
      dtype=torch.float32,
    )

  original_geom_pos = model.geom_pos.copy()

  sync_model_fields(model, _SimModel(), {"geom_rgba"}, env_idx=1)

  np.testing.assert_allclose(model.geom_rgba, [[0.5, 0.6, 0.7, 0.8]])
  np.testing.assert_allclose(model.geom_pos, original_geom_pos)
