"""Tests for spec.py utilities."""

import mujoco
import numpy as np
import pytest

from mjlab.utils.spec import create_position_actuator, create_velocity_actuator


@pytest.fixture
def spec_with_limited_joint():
  """Create a spec with a limited joint for testing."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="test_body")
  joint = body.add_joint(
    name="test_joint",
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=[0, 0, 1],
    range=[-1.0, 1.0],
  )
  joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], mass=1.0)
  return spec


def test_position_actuator_allows_setpoints_beyond_joint_limits(
  spec_with_limited_joint,
):
  """Verify that position actuators allow commanding setpoints beyond joint limits.

  This is the core fix: position actuators have ctrllimited=False, allowing
  policies to command positions outside kinematic limits to maximize torque.
  We verify this by comparing torque when commanding beyond vs at the limit.
  """
  stiffness = 100.0
  create_position_actuator(
    spec_with_limited_joint,
    "test_joint",
    stiffness=stiffness,
    damping=10.0,
    effort_limit=500.0,
  )
  model = spec_with_limited_joint.compile()
  data = mujoco.MjData(model)

  # Verify ctrllimited=False.
  actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "test_joint")
  assert model.actuator_ctrllimited[actuator_id] == 0

  # Set joint position.
  current_pos = 0.5
  data.qpos[0] = current_pos
  data.qvel[0] = 0.0

  # Command beyond joint upper limit (1.0).
  target_beyond = 2.0
  data.ctrl[0] = target_beyond
  mujoco.mj_forward(model, data)
  torque_beyond = data.actuator_force[0]

  # Expected: -kp * (q - target) = -100 * (0.5 - 2.0) = 150.0.
  expected_torque_beyond = -stiffness * (current_pos - target_beyond)
  np.testing.assert_allclose(torque_beyond, expected_torque_beyond, rtol=1e-5)

  # If control were clipped to limit (1.0), torque would be 50.0.
  expected_if_clipped = -stiffness * (current_pos - 1.0)
  assert abs(torque_beyond) == 3.0 * abs(expected_if_clipped)


def test_position_actuator_forces_clipped_to_effort_limit(spec_with_limited_joint):
  """Verify that actuator forces are clipped to effort_limit, not control limits.

  While control setpoints aren't clipped, forces must still be limited to
  prevent unrealistic torques.
  """
  effort_limit = 10.0
  stiffness = 1000.0
  create_position_actuator(
    spec_with_limited_joint,
    "test_joint",
    stiffness=stiffness,
    damping=1.0,
    effort_limit=effort_limit,
  )
  model = spec_with_limited_joint.compile()
  data = mujoco.MjData(model)

  # Verify forcelimited=True.
  actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "test_joint")
  assert model.actuator_forcelimited[actuator_id] == 1
  np.testing.assert_array_almost_equal(
    model.actuator_forcerange[actuator_id], [-effort_limit, effort_limit]
  )

  # Large position error would generate force > effort_limit without clipping.
  data.qpos[0] = 0.0
  data.ctrl[0] = 3.0
  mujoco.mj_step(model, data)

  # Force should saturate at effort_limit.
  assert abs(data.actuator_force[0]) <= effort_limit + 1e-6
  assert abs(data.actuator_force[0]) >= effort_limit - 1e-3


def test_ctrllimited_true_would_clip_internally():
  """Demonstrate that ctrllimited=True clips control internally (the bug).

  This test shows what would happen with the bug: commanding at vs beyond
  the limit generates identical forces, proving internal clipping.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="test_body")
  joint = body.add_joint(
    name="test_joint",
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=[0, 0, 1],
    range=[-1.0, 1.0],
  )
  joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], mass=1.0)

  # Manually create actuator WITH THE BUG: ctrllimited=True.
  stiffness = 100.0
  actuator = spec.add_actuator(name="test_joint", target="test_joint")
  actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
  actuator.ctrllimited = True  # THE BUG.
  actuator.ctrlrange[:] = np.array([-1.0, 1.0])
  actuator.gainprm[0] = stiffness
  actuator.biasprm[1] = -stiffness
  actuator.biasprm[2] = -10.0

  model = spec.compile()
  data = mujoco.MjData(model)

  data.qpos[0] = 0.5
  data.qvel[0] = 0.0

  # Command at limit.
  data.ctrl[0] = 1.0
  mujoco.mj_forward(model, data)
  force_at_limit = data.actuator_force[0]

  # Command beyond limit.
  data.ctrl[0] = 2.0
  mujoco.mj_forward(model, data)
  force_beyond = data.actuator_force[0]

  # With the bug, forces are IDENTICAL (both clipped to 1.0 internally).
  np.testing.assert_allclose(force_beyond, force_at_limit, rtol=1e-10)
  expected_force = -stiffness * (0.5 - 1.0)
  np.testing.assert_allclose(force_at_limit, expected_force, rtol=1e-5)


@pytest.fixture
def spec_with_continuous_joint():
  """Create a spec with a continuous (unlimited) hinge joint."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="wheel")
  body.add_joint(
    name="wheel_joint",
    type=mujoco.mjtJoint.mjJNT_HINGE,
    axis=[0, 0, 1],
  )
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.1, 0.05], mass=1.0)
  return spec


def test_velocity_actuator_continuous_joint(spec_with_continuous_joint):
  """Velocity actuator compiles for continuous joints with no range."""
  damping = 10.0
  create_velocity_actuator(
    spec_with_continuous_joint,
    "wheel_joint",
    damping=damping,
  )
  model = spec_with_continuous_joint.compile()
  data = mujoco.MjData(model)

  actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_joint")
  assert model.actuator_ctrllimited[actuator_id] == 0

  # Commanding a velocity target should produce force = damping * (ctrl - qvel).
  data.qvel[0] = 0.0
  data.ctrl[0] = 5.0
  mujoco.mj_forward(model, data)
  expected_force = damping * 5.0
  np.testing.assert_allclose(data.actuator_force[0], expected_force, rtol=1e-5)


def test_velocity_actuator_forces_clipped_to_effort_limit(
  spec_with_continuous_joint,
):
  """Velocity actuator force is bounded by effort_limit."""
  effort_limit = 20.0
  create_velocity_actuator(
    spec_with_continuous_joint,
    "wheel_joint",
    damping=1000.0,
    effort_limit=effort_limit,
  )
  model = spec_with_continuous_joint.compile()
  data = mujoco.MjData(model)

  actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_joint")
  assert model.actuator_forcelimited[actuator_id] == 1

  # Large velocity error would exceed effort_limit without clamping.
  data.qvel[0] = 0.0
  data.ctrl[0] = 100.0
  mujoco.mj_forward(model, data)
  assert abs(data.actuator_force[0]) <= effort_limit + 1e-6
