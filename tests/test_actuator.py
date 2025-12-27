"""Tests for actuator module."""

import pytest
import torch
from conftest import (
  create_entity_with_actuator,
  get_test_device,
  initialize_entity,
  load_fixture_xml,
)

from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  IdealPdActuatorCfg,
)


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def robot_xml():
  return load_fixture_xml("floating_base_articulated")


def test_builtin_pd_actuator_compute(device, robot_xml):
  """BuiltinPositionActuator writes position targets to ctrl."""
  actuator_cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint.*",), stiffness=50.0, damping=5.0
  )
  entity = create_entity_with_actuator(robot_xml, actuator_cfg)
  entity, sim = initialize_entity(entity, device)

  entity.set_joint_position_target(torch.tensor([[0.5, -0.3]], device=device))
  entity.write_data_to_sim()

  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([0.5, -0.3], device=device))


def test_ideal_pd_actuator_compute(device, robot_xml):
  """IdealPdActuator computes torques via explicit PD control."""
  actuator_cfg = IdealPdActuatorCfg(
    target_names_expr=("joint.*",), effort_limit=100.0, stiffness=50.0, damping=5.0
  )
  entity = create_entity_with_actuator(robot_xml, actuator_cfg)
  entity, sim = initialize_entity(entity, device)

  entity.write_joint_state_to_sim(
    position=torch.tensor([[0.0, 0.0]], device=device),
    velocity=torch.tensor([[0.0, 0.0]], device=device),
  )

  entity.set_joint_position_target(torch.tensor([[0.1, -0.1]], device=device))
  entity.set_joint_velocity_target(torch.tensor([[0.0, 0.0]], device=device))
  entity.set_joint_effort_target(torch.tensor([[0.0, 0.0]], device=device))
  entity.write_data_to_sim()

  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([5.0, -5.0], device=device))


def test_targets_cleared_on_reset(device, robot_xml):
  """Entity.reset() zeros all targets."""
  actuator_cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint.*",), stiffness=50.0, damping=5.0
  )
  entity = create_entity_with_actuator(robot_xml, actuator_cfg)
  entity, sim = initialize_entity(entity, device)

  entity.set_joint_position_target(torch.tensor([[0.5, -0.3]], device=device))
  entity.write_data_to_sim()

  assert not torch.allclose(
    entity.data.joint_pos_target, torch.zeros(1, 2, device=device)
  )

  entity.reset()

  assert torch.allclose(entity.data.joint_pos_target, torch.zeros(1, 2, device=device))
  assert torch.allclose(entity.data.joint_vel_target, torch.zeros(1, 2, device=device))
  assert torch.allclose(
    entity.data.joint_effort_target, torch.zeros(1, 2, device=device)
  )
