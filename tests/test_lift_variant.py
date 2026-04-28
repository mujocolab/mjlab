"""Tier 1 tests for the yam lift-variant scaffolding.

Verifies each variant spec compiles cleanly and produces mass and inertia
within tolerance of analytic primitive formulas.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from mjlab.tasks.manipulation.config.yam.env_cfgs import (
  LIFT_VARIANT_DEFAULTS,
  make_box_variant_spec,
  make_ellipsoid_variant_spec,
  make_sphere_variant_spec,
)

# YAM linear gripper stroke is 0.071 m (see GRIPPER_LINEAR_STROKE_CRANK in
# i2rt_yam.yam_constants). A parallel-jaw grasp engages across the object's
# smallest axis, so we check the smallest AABB dimension fits the stroke
# with a small margin for finger thickness.
_GRIPPER_OPENING = 0.065  # m, slightly less than the 0.071 m mechanical stroke.

# Density used by all default variants.
_DEFAULT_DENSITY = 300.0


def _compile_object_body(spec: mujoco.MjSpec) -> tuple[mujoco.MjModel, int]:
  """Compile *spec* and return (model, body_id) for the 'cube' body."""
  model = spec.compile()
  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
  assert body_id >= 0, "spec must have a body named 'cube'"
  return model, body_id


def _box_inertia_eigenvalues(
  mass: float, half_extents: tuple[float, float, float]
) -> np.ndarray:
  sx, sy, sz = half_extents
  return np.sort(
    np.array(
      [
        mass * (sy**2 + sz**2) / 3,
        mass * (sx**2 + sz**2) / 3,
        mass * (sx**2 + sy**2) / 3,
      ]
    )
  )


def _ellipsoid_inertia_eigenvalues(
  mass: float, semi_axes: tuple[float, float, float]
) -> np.ndarray:
  a, b, c = semi_axes
  return np.sort(
    np.array(
      [
        mass * (b**2 + c**2) / 5,
        mass * (a**2 + c**2) / 5,
        mass * (a**2 + b**2) / 5,
      ]
    )
  )


def test_default_variant_set_is_six():
  """Default set is the 2x3 grid of (sharp/smooth) x (isotropic/long/flat)."""
  assert len(LIFT_VARIANT_DEFAULTS) == 6
  expected = {"cube", "box_long", "box_flat", "sphere", "pencil", "plate"}
  assert set(LIFT_VARIANT_DEFAULTS.keys()) == expected


def test_all_default_variants_compile():
  """Each variant spec compiles and has the expected single-body topology."""
  for name, vcfg in LIFT_VARIANT_DEFAULTS.items():
    spec = vcfg.spec_fn()
    model, body_id = _compile_object_body(spec)
    # Exactly one freejoint on the object body.
    assert model.nbody == 2, f"{name}: expected 1 worldbody + 1 object body"
    assert model.njnt == 1, f"{name}: expected 1 joint, got {model.njnt}"
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    # Exactly one mesh geom on the object body.
    body_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
    assert len(body_geoms) == 1, f"{name}: expected 1 geom, got {len(body_geoms)}"
    assert model.geom_type[body_geoms[0]] == mujoco.mjtGeom.mjGEOM_MESH


@pytest.mark.parametrize(
  "half_extents",
  [(0.030, 0.030, 0.030), (0.050, 0.015, 0.015), (0.035, 0.035, 0.012)],
  ids=["cube", "box_long", "box_flat"],
)
def test_box_variant_mass_and_inertia(half_extents: tuple[float, float, float]):
  """Box mass = density * 8*sx*sy*sz; inertia matches analytic to FP precision."""
  spec = make_box_variant_spec(half_extents, density=_DEFAULT_DENSITY)
  model, body_id = _compile_object_body(spec)

  expected_mass = (
    _DEFAULT_DENSITY * 8 * half_extents[0] * half_extents[1] * half_extents[2]
  )
  np.testing.assert_allclose(model.body_mass[body_id], expected_mass, rtol=1e-4)

  expected_eigs = _box_inertia_eigenvalues(expected_mass, half_extents)
  actual_eigs = np.sort(model.body_inertia[body_id])
  np.testing.assert_allclose(actual_eigs, expected_eigs, rtol=1e-3)


def test_sphere_variant_mass_and_inertia():
  """Sphere mass and inertia approximate analytic within icosphere tolerance."""
  radius = 0.025
  spec = make_sphere_variant_spec(radius, density=_DEFAULT_DENSITY)
  model, body_id = _compile_object_body(spec)

  expected_mass = _DEFAULT_DENSITY * (4.0 / 3.0) * math.pi * radius**3
  np.testing.assert_allclose(model.body_mass[body_id], expected_mass, rtol=0.05)

  expected_I = (2.0 / 5.0) * model.body_mass[body_id] * radius**2
  np.testing.assert_allclose(model.body_inertia[body_id], expected_I, rtol=0.10)


@pytest.mark.parametrize(
  "semi_axes",
  [(0.050, 0.020, 0.020), (0.035, 0.035, 0.018)],
  ids=["pencil", "plate"],
)
def test_ellipsoid_variant_mass_and_inertia(semi_axes: tuple[float, float, float]):
  """Ellipsoid mass and principal inertia approximate analytic within icosphere tolerance."""
  spec = make_ellipsoid_variant_spec(semi_axes, density=_DEFAULT_DENSITY)
  model, body_id = _compile_object_body(spec)

  a, b, c = semi_axes
  expected_mass = _DEFAULT_DENSITY * (4.0 / 3.0) * math.pi * a * b * c
  np.testing.assert_allclose(model.body_mass[body_id], expected_mass, rtol=0.05)

  # MuJoCo reports body_inertia in the principal-axis frame (with body_iquat
  # rotating to body coords); compare sorted eigenvalues.
  expected_eigs = _ellipsoid_inertia_eigenvalues(model.body_mass[body_id], semi_axes)
  actual_eigs = np.sort(model.body_inertia[body_id])
  np.testing.assert_allclose(actual_eigs, expected_eigs, rtol=0.15)


def test_box_mesh_face_winding_is_outward():
  """All box mesh triangles must wind so normals point outward.

  Inverted normals make faces invisible under back-face culling in viser /
  three.js while still giving correct mass and inertia (the volume integral
  takes the absolute value).
  """
  spec = make_box_variant_spec((0.03, 0.02, 0.025))
  model, _ = _compile_object_body(spec)
  # The compiled model has one mesh; pull its verts and faces.
  verts = model.mesh_vert.reshape(-1, 3)
  faces = model.mesh_face.reshape(-1, 3)
  for tri in faces:
    v = verts[tri]
    normal = np.cross(v[1] - v[0], v[2] - v[0])
    centroid = v.mean(axis=0)
    # For a convex mesh centered at origin, an outward-winding triangle has
    # a positive dot product between its normal and the centroid.
    assert float(np.dot(normal, centroid)) > 0, f"Triangle {tri.tolist()} winds inward."


def test_all_default_variants_fit_gripper():
  """Min AABB dimension of every variant fits within the gripper opening."""
  for name, vcfg in LIFT_VARIANT_DEFAULTS.items():
    spec = vcfg.spec_fn()
    model, body_id = _compile_object_body(spec)
    geom_id = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id)
    # geom_aabb is (center, half_extent) for the geom in body-local frame.
    half_extents = model.geom_aabb[geom_id].reshape(2, 3)[1]
    min_width = 2 * float(half_extents.min())
    assert min_width <= _GRIPPER_OPENING, (
      f"{name}: min AABB width {min_width:.4f} m exceeds gripper opening "
      f"{_GRIPPER_OPENING:.4f} m."
    )


def test_lift_variant_env_builds():
  """Smoke: full env constructs with the variant set on a small num_envs."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.manipulation.config.yam.env_cfgs import yam_lift_variant_env_cfg

  cfg = yam_lift_variant_env_cfg(play=True)
  cfg.scene.num_envs = len(LIFT_VARIANT_DEFAULTS)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    sim = env.unwrapped.sim
    # Per-world mesh path is active.
    assert "geom_dataid" in sim.expanded_fields
    assert "body_mass" in sim.per_world_default_fields
    # Each variant gets at least one world.
    w2v = sim._wp_model.geom_dataid.numpy()
    assert w2v.shape[0] == cfg.scene.num_envs
  finally:
    env.close()
