"""Tests for go1_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_go1 import go1_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def go1_entity() -> Entity:
  return Entity(go1_constants.get_go1_robot_cfg())


@pytest.fixture(scope="module")
def go1_model(go1_entity: Entity) -> mujoco.MjModel:
  return go1_entity.spec.compile()


# fmt: off
@pytest.mark.parametrize(
  "actuator_config,stiffness,damping",
  [
    (go1_constants.GO1_HIP_ACTUATOR_CFG, go1_constants.STIFFNESS_HIP, go1_constants.DAMPING_HIP),
    (go1_constants.GO1_KNEE_ACTUATOR_CFG, go1_constants.STIFFNESS_KNEE, go1_constants.DAMPING_KNEE),
  ],
)
# fmt: on
def test_actuator_parameters(go1_model, actuator_config, stiffness, damping):
  for i in range(go1_model.nu):
    actuator = go1_model.actuator(i)
    actuator_name = actuator.name
    matches = any(
      re.match(pattern, actuator_name) for pattern in actuator_config.joint_names_expr
    )
    if matches:
      assert actuator.gainprm[0] == stiffness
      assert actuator.biasprm[1] == -stiffness
      assert actuator.biasprm[2] == -damping
      assert actuator.forcerange[0] == -actuator_config.effort_limit
      assert actuator.forcerange[1] == actuator_config.effort_limit


def test_keyframe_joint_positions(go1_entity, go1_model) -> None:
  """Test that keyframe joint positions match the configuration."""
  key = go1_model.key("init_state")
  expected_joint_pos = go1_constants.INIT_STATE.joint_pos
  expected_values = resolve_expr(expected_joint_pos, go1_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    go1_entity.joint_names, expected_values, strict=True
  ):
    joint = go1_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: "
      f"expected {expected_value}, got {actual_value}",
    )


def test_foot_collision_geoms(go1_model) -> None:
  """Foot collision geoms should have specific properties."""
  foot_pattern = r"^[FR][LR]_foot_collision$"
  for i in range(go1_model.ngeom):
    geom = go1_model.geom(i)
    if re.match(foot_pattern, geom.name):
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6


def test_collision_geom_count(go1_model) -> None:
  """Go1 should have 4 foot collision geoms."""
  foot_pattern = r"^[FR][LR]_foot_collision$"
  foot_geoms = [
    g.name for g in [go1_model.geom(i) for i in range(go1_model.ngeom)]
    if re.match(foot_pattern, g.name)
  ]
  assert len(foot_geoms) == 4


def test_go1_entity_creation(go1_entity) -> None:
  """Test basic Go1 entity properties."""
  assert go1_entity.num_actuators == 12
  assert go1_entity.num_joints == 12
  assert go1_entity.is_actuated
  assert not go1_entity.is_fixed_base
