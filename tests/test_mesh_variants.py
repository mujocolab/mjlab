"""Tests for per-world mesh variant support."""

from __future__ import annotations

from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import EntityCfg, VariantCfg, VariantEntityCfg
from mjlab.sim.mesh_variants import allocate_worlds, build_mesh_variant_model
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


# Entity merging.


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


# Validation.


def test_mismatched_joint_structure_raises():
  cfg = VariantEntityCfg(
    variants={
      "sphere": VariantCfg(spec_fn=_simple_sphere_spec, weight=0.5),
      "hinge": VariantCfg(spec_fn=_hinge_spec, weight=0.5),
    },
  )
  with pytest.raises(ValueError, match="joint"):
    cfg.build()


def test_single_variant_builds():
  """A single variant degenerates cleanly; useful for templated variant sets."""
  cfg = VariantEntityCfg(
    variants={"only": VariantCfg(spec_fn=_simple_sphere_spec)},
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
      "a": VariantCfg(spec_fn=_fixed_base_sphere_spec, weight=0.5),
      "b": VariantCfg(spec_fn=_fixed_base_sphere_spec, weight=0.5),
    },
  )
  with pytest.raises(ValueError, match="floating-base"):
    cfg.build()


def test_setting_spec_fn_on_variant_cfg_raises():
  """VariantEntityCfg.spec_fn is unused; setting it should fail loudly."""
  with pytest.raises(ValueError, match="spec_fn cannot be set"):
    VariantEntityCfg(
      variants={"only": VariantCfg(spec_fn=_simple_sphere_spec)},
      spec_fn=_simple_sphere_spec,
    )


def test_no_variants_unchanged():
  cfg = EntityCfg(spec_fn=_simple_sphere_spec)
  entity = cfg.build()
  assert entity.variant_metadata is None


# build_mesh_variant_model: dataid and dependent fields.


def test_dataid_assigned_per_world():
  """Each world's geom_dataid points to its variant's meshes."""
  scene_spec, vi = _build_scene_with_variants(_simple_sphere_spec, _simple_cone_spec)
  result = build_mesh_variant_model(scene_spec, 4, vi)

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
  result = build_mesh_variant_model(scene_spec, 4, vi)

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
  result = build_mesh_variant_model(scene_spec, 4, vi)

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
      "sphere": VariantCfg(
        lambda: _explicit_variant("sphere", 0.2, (1e-4, 2e-4, 3e-4)),
        weight=0.5,
      ),
      "cone": VariantCfg(
        lambda: _explicit_variant("cone", 0.7, (4e-4, 5e-4, 6e-4), cone=True),
        weight=0.5,
      ),
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
            "sphere": VariantCfg(_simple_sphere_spec, weight=0.5),
            "cone": VariantCfg(_simple_cone_spec, weight=0.5),
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
            "sphere": VariantCfg(_simple_sphere_spec, weight=0.5),
            "cone": VariantCfg(_simple_cone_spec, weight=0.5),
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
      "light": VariantCfg(
        lambda: _explicit_mass_variant("light", light_mass), weight=0.5
      ),
      "heavy": VariantCfg(
        lambda: _explicit_mass_variant("heavy", heavy_mass, cone=True), weight=0.5
      ),
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
