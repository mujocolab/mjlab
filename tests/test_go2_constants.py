"""Tests for go2_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_go2 import go2_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def go2_entity() -> Entity:
  return Entity(go2_constants.get_go2_robot_cfg())


@pytest.fixture(scope="module")
def go2_model(go2_entity: Entity) -> mujoco.MjModel:
  return go2_entity.spec.compile()


# fmt: off
@pytest.mark.parametrize(
  "actuator_config,stiffness,damping",
  [
    (
      go2_constants.GO2_HIP_ACTUATOR_CFG,
      go2_constants.STIFFNESS_HIP,
      go2_constants.DAMPING_HIP,
    ),
    (
      go2_constants.GO2_KNEE_ACTUATOR_CFG,
      go2_constants.STIFFNESS_KNEE,
      go2_constants.DAMPING_KNEE,
    ),
  ],
)
# fmt: on
def test_actuator_parameters(go2_model, actuator_config, stiffness, damping):
  for i in range(go2_model.nu):
    actuator = go2_model.actuator(i)
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


def test_keyframe_joint_positions(go2_entity, go2_model) -> None:
  """Test that keyframe joint positions match the configuration."""
  key = go2_model.key("init_state")
  expected_joint_pos = go2_constants.INIT_STATE.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, go2_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    go2_entity.joint_names, expected_values, strict=True
  ):
    joint = go2_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: "
      f"expected {expected_value}, got {actual_value}",
    )


def test_foot_collision_geoms(go2_model) -> None:
  """Foot collision geoms should have specific properties."""
  foot_pattern = r"^[FR][LR]_foot_collision$"
  for i in range(go2_model.ngeom):
    geom = go2_model.geom(i)
    if re.match(foot_pattern, geom.name):
      assert geom.condim == 6
      assert geom.priority == 1
      assert geom.friction[0] == 0.8


def test_collision_geom_count(go2_model) -> None:
  """Go2 should have 4 foot collision geoms."""
  foot_pattern = r"^[FR][LR]_foot_collision$"
  foot_geoms = [
    g.name for g in [go2_model.geom(i) for i in range(go2_model.ngeom)]
    if re.match(foot_pattern, g.name)
  ]
  assert len(foot_geoms) == 4


def test_go2_entity_creation(go2_entity) -> None:
  """Test basic Go2 entity properties."""
  assert go2_entity.num_actuators == 12
  assert go2_entity.num_joints == 12
  assert go2_entity.is_actuated
  assert not go2_entity.is_fixed_base
