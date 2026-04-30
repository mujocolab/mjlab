"""Tests for h1_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_h1 import h1_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def h1_entity() -> Entity:
  return Entity(h1_constants.get_h1_robot_cfg())


@pytest.fixture(scope="module")
def h1_model(h1_entity: Entity) -> mujoco.MjModel:
  return h1_entity.spec.compile()


# fmt: off
@pytest.mark.parametrize(
  "actuator_config,stiffness,damping",
  [
    (h1_constants.H1_ACTUATOR_HIP, 200.0, 5.0),
    (h1_constants.H1_ACTUATOR_KNEE, 300.0, 6.0),
    (h1_constants.H1_ACTUATOR_ANKLE, 40.0, 2.0),
    (h1_constants.H1_ACTUATOR_TORSO, 300.0, 6.0),
    (h1_constants.H1_ACTUATOR_SHOULDER_PITCH_ROLL, 20.0, 0.5),
    (h1_constants.H1_ACTUATOR_SHOULDER_YAW_ELBOW, 20.0, 0.5),
  ],
)
# fmt: on
def test_actuator_parameters(h1_model, actuator_config, stiffness, damping):
  for i in range(h1_model.nu):
    actuator = h1_model.actuator(i)
    actuator_name = actuator.name
    matches = any(
      re.match(pattern, actuator_name) for pattern in actuator_config.target_names_expr
    )
    if matches:
      assert actuator.gainprm[0] == stiffness
      assert actuator.biasprm[1] == -stiffness
      assert actuator.biasprm[2] == -damping
      assert actuator.forcerange[0] == -actuator_config.effort_limit
      assert actuator.forcerange[1] == actuator_config.effort_limit


def test_keyframe_base_position(h1_model) -> None:
  data = mujoco.MjData(h1_model)
  mujoco.mj_resetDataKeyframe(h1_model, data, 0)
  mujoco.mj_forward(h1_model, data)
  np.testing.assert_array_equal(data.qpos[:3], h1_constants.KNEES_BENT_KEYFRAME.pos)
  np.testing.assert_array_equal(data.qpos[3:7], h1_constants.KNEES_BENT_KEYFRAME.rot)


def test_keyframe_joint_positions(h1_entity, h1_model) -> None:
  """Test that keyframe joint positions match the configuration."""
  key = h1_model.key("init_state")
  expected_joint_pos = h1_constants.KNEES_BENT_KEYFRAME.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, h1_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    h1_entity.joint_names, expected_values, strict=True
  ):
    joint = h1_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: "
      f"expected {expected_value}, got {actual_value}",
    )


def test_foot_collision_geoms(h1_model) -> None:
  foot_pattern = r"^(left|right)_foot[1-2]_collision$"
  for i in range(h1_model.ngeom):
    geom = h1_model.geom(i)
    geom_name = geom.name
    # Foot collision geoms should have condim=3, priority=1, and friction=0.6.
    if re.match(foot_pattern, geom_name):
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6


def test_non_foot_collision_geoms(h1_model) -> None:
  foot_pattern = r"^(left|right)_foot[1-2]_collision$"
  for i in range(h1_model.ngeom):
    geom = h1_model.geom(i)
    geom_name = geom.name
    if "_collision" not in geom_name:
      continue
    # Non-foot collision geoms should have condim=1
    if not re.match(foot_pattern, geom_name):
      assert geom.condim == 1


def test_collision_geom_count(h1_model) -> None:
  # There should be 2 geoms (capsules) per foot.
  collision_geoms = [
    h1_model.geom(i).name
    for i in range(h1_model.ngeom)
    if "_collision" in h1_model.geom(i).name
  ]
  assert len(collision_geoms) > 0
  foot_geoms = [
    name
    for name in collision_geoms
    if re.match(r"^(left|right)_foot[1-2]_collision$", name)
  ]
  assert len(foot_geoms) == 4


def test_h1_entity_creation(h1_entity) -> None:
  assert h1_entity.num_actuators == 19
  assert h1_entity.num_joints == 19
  assert h1_entity.is_actuated
  assert not h1_entity.is_fixed_base


def test_h1_actuators_configured_correctly(h1_model):
  """Verify that all H1 actuators have correct control and force limiting.

  All 19 H1 actuators should have ctrllimited=False (allowing setpoints beyond
  joint limits) and forcelimited=True (limiting forces to effort limits).
  """
  for i in range(h1_model.nu):
    actuator_name = h1_model.actuator(i).name
    assert h1_model.actuator_ctrllimited[i] == 0, (
      f"Actuator '{actuator_name}' has ctrllimited=True, expected False"
    )
    assert h1_model.actuator_forcelimited[i] == 1, (
      f"Actuator '{actuator_name}' has forcelimited=False, expected True"
    )
