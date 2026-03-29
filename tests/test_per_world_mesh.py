"""Tests for per-world mesh variant support."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mjlab.entity import EntityCfg, VariantCfg, VariantEntityCfg
from mjlab.scene.per_world_mesh import allocate_worlds
from mjlab.viewer.native.viewer import _disable_model_sameframe_shortcuts

# ---------------------------------------------------------------------------
# Helpers: variant specs with visual + collision mesh geoms.
# ---------------------------------------------------------------------------


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
    variants={
      "a": VariantCfg(spec_fn=variant_a_fn, weight=weight_a),
      "b": VariantCfg(spec_fn=variant_b_fn, weight=weight_b),
    },
  )
  entity = cfg.build()
  assert entity.variant_metadata is not None
  scene_spec = mujoco.MjSpec()
  frame = scene_spec.worldbody.add_frame()
  scene_spec.attach(entity.spec, prefix="object/", frame=frame)
  return scene_spec, [("object/", entity.variant_metadata)]


# ---------------------------------------------------------------------------
# allocate_worlds
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Entity merging
# ---------------------------------------------------------------------------


def test_entity_builds_with_variants():
  cfg = VariantEntityCfg(
    variants={
      "sphere": VariantCfg(spec_fn=_simple_sphere_spec, weight=0.5),
      "cone": VariantCfg(spec_fn=_simple_cone_spec, weight=0.5),
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
      "sphere": VariantCfg(spec_fn=_sphere_2col_spec, weight=0.5),
      "cone": VariantCfg(spec_fn=_cone_4col_spec, weight=0.5),
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mismatched_joint_structure_raises():
  cfg = VariantEntityCfg(
    variants={
      "sphere": VariantCfg(spec_fn=_simple_sphere_spec, weight=0.5),
      "hinge": VariantCfg(spec_fn=_hinge_spec, weight=0.5),
    },
  )
  with pytest.raises(ValueError, match="joint"):
    cfg.build()


def test_single_variant_raises():
  cfg = VariantEntityCfg(
    variants={"only": VariantCfg(spec_fn=_simple_sphere_spec)},
  )
  with pytest.raises(ValueError, match="at least 2"):
    cfg.build()


def test_no_variants_unchanged():
  cfg = EntityCfg(spec_fn=_simple_sphere_spec)
  entity = cfg.build()
  assert entity.variant_metadata is None


# ---------------------------------------------------------------------------
# per_world_mesh: dataid and dependent fields
# ---------------------------------------------------------------------------


def test_dataid_assigned_per_world():
  """Each world's geom_dataid points to its variant's meshes."""
  from mjlab.scene.per_world_mesh import per_world_mesh

  scene_spec, vi = _build_scene_with_variants(_simple_sphere_spec, _simple_cone_spec)
  result = per_world_mesh(scene_spec, 4, vi)

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
  from mjlab.scene.per_world_mesh import per_world_mesh

  scene_spec, vi = _build_scene_with_variants(_sphere_2col_spec, _cone_4col_spec)
  result = per_world_mesh(scene_spec, 4, vi)

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
  # First 3 should be valid (>= 0).
  assert all(d >= 0 for d in sphere_dataid[:3])


def test_dependent_fields_match_individual_compilation():
  """Per-world body_mass matches independently compiled variant models."""
  from mjlab.scene.per_world_mesh import per_world_mesh

  scene_spec, vi = _build_scene_with_variants(_simple_sphere_spec, _simple_cone_spec)
  result = per_world_mesh(scene_spec, 4, vi)

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


# ---------------------------------------------------------------------------
# Full env lifecycle
# ---------------------------------------------------------------------------


def test_env_step_with_variants():
  """Build a full ManagerBasedRlEnv with variants; step without crashing."""
  import torch

  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.envs.mdp.events import reset_root_state_uniform
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.scene import SceneCfg
  from mjlab.terrains import TerrainEntityCfg

  object_cfg = VariantEntityCfg(
    variants={
      "sphere": VariantCfg(_simple_sphere_spec, weight=0.5),
      "cone": VariantCfg(_simple_cone_spec, weight=0.5),
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


# ---------------------------------------------------------------------------
# Viewer: sameframe shortcut fix
# ---------------------------------------------------------------------------


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
  _disable_model_sameframe_shortcuts(base_model)
  mujoco.mj_forward(base_model, base_data)
  np.testing.assert_allclose(base_data.geom_xpos, cone_data.geom_xpos, atol=1e-6)
