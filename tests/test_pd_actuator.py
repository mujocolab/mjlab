"""Tests for PD actuator equivalence."""

import pytest
import torch
from conftest import (
  create_entity_with_actuator,
  get_test_device,
  initialize_entity,
  load_fixture_xml,
)

from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def robot_xml():
  return load_fixture_xml("floating_base_articulated")


def test_ideal_pd_matches_builtin_at_rest(device, robot_xml):
  """IdealPd and BuiltinPd produce equivalent actuator forces (qfrc_actuator)."""
  kp, kv = 80.0, 10.0

  ideal_entity = create_entity_with_actuator(
    robot_xml,
    IdealPdActuatorCfg(
      target_names_expr=("joint.*",), effort_limit=100.0, stiffness=kp, damping=kv
    ),
  )
  builtin_entity = create_entity_with_actuator(
    robot_xml,
    BuiltinPositionActuatorCfg(
      target_names_expr=("joint.*",), effort_limit=100.0, stiffness=kp, damping=kv
    ),
  )

  ideal_entity, ideal_sim = initialize_entity(ideal_entity, device)
  builtin_entity, builtin_sim = initialize_entity(builtin_entity, device)

  joint_pos = torch.tensor([[0.1, -0.05]], device=device)
  joint_vel = torch.tensor([[0.0, 0.0]], device=device)
  ideal_entity.write_joint_state_to_sim(joint_pos, joint_vel)
  builtin_entity.write_joint_state_to_sim(joint_pos, joint_vel)

  pos_target = torch.tensor([[0.5, -0.3]], device=device)
  vel_target = torch.tensor([[0.2, -0.1]], device=device)
  ideal_entity.set_joint_position_target(pos_target)
  ideal_entity.set_joint_velocity_target(vel_target)
  ideal_entity.set_joint_effort_target(torch.zeros(1, 2, device=device))
  builtin_entity.set_joint_position_target(pos_target)

  ideal_entity.write_data_to_sim()
  builtin_entity.write_data_to_sim()

  joint_v_adr = ideal_entity.indexing.joint_v_adr
  ideal_qfrc = ideal_sim.data.qfrc_actuator[0, joint_v_adr]
  builtin_qfrc = builtin_sim.data.qfrc_actuator[0, joint_v_adr]

  assert torch.allclose(ideal_qfrc, builtin_qfrc, rtol=1e-4, atol=1e-5)


def test_ideal_pd_matches_builtin_with_velocity(device, robot_xml):
  """IdealPd and BuiltinPd produce equivalent damping forces with nonzero vel."""
  kp, kv = 80.0, 10.0

  ideal_entity = create_entity_with_actuator(
    robot_xml,
    IdealPdActuatorCfg(
      target_names_expr=("joint.*",), effort_limit=100.0, stiffness=kp, damping=kv
    ),
  )
  builtin_entity = create_entity_with_actuator(
    robot_xml,
    BuiltinPositionActuatorCfg(
      target_names_expr=("joint.*",), effort_limit=100.0, stiffness=kp, damping=kv
    ),
  )

  ideal_entity, ideal_sim = initialize_entity(ideal_entity, device)
  builtin_entity, builtin_sim = initialize_entity(builtin_entity, device)

  joint_pos = torch.tensor([[0.2, -0.1]], device=device)
  joint_vel = torch.tensor([[0.8, -0.5]], device=device)
  ideal_entity.write_joint_state_to_sim(joint_pos, joint_vel)
  builtin_entity.write_joint_state_to_sim(joint_pos, joint_vel)

  pos_target = torch.tensor([[0.5, 0.1]], device=device)
  vel_target = torch.tensor([[0.3, -0.2]], device=device)
  ideal_entity.set_joint_position_target(pos_target)
  ideal_entity.set_joint_velocity_target(vel_target)
  ideal_entity.set_joint_effort_target(torch.zeros(1, 2, device=device))
  builtin_entity.set_joint_position_target(pos_target)

  ideal_entity.write_data_to_sim()
  builtin_entity.write_data_to_sim()

  joint_v_adr = ideal_entity.indexing.joint_v_adr
  ideal_qfrc = ideal_sim.data.qfrc_actuator[0, joint_v_adr]
  builtin_qfrc = builtin_sim.data.qfrc_actuator[0, joint_v_adr]

  assert torch.allclose(ideal_qfrc, builtin_qfrc, rtol=1e-4, atol=1e-5)


def test_ideal_pd_with_feedforward_effort(device, robot_xml):
  """IdealPd adds feedforward effort to PD output."""
  kp, kv = 50.0, 5.0

  ideal_entity = create_entity_with_actuator(
    robot_xml,
    IdealPdActuatorCfg(
      target_names_expr=("joint.*",), effort_limit=100.0, stiffness=kp, damping=kv
    ),
  )

  ideal_entity, ideal_sim = initialize_entity(ideal_entity, device)

  joint_pos = torch.tensor([[0.5, 0.0]], device=device)
  joint_vel = torch.tensor([[0.0, 0.0]], device=device)
  ideal_entity.write_joint_state_to_sim(joint_pos, joint_vel)

  ideal_entity.set_joint_position_target(joint_pos)
  ideal_entity.set_joint_velocity_target(joint_vel)
  ideal_entity.set_joint_effort_target(torch.tensor([[2.0, -1.0]], device=device))
  ideal_entity.write_data_to_sim()

  ideal_ctrl = ideal_sim.data.ctrl[0]
  assert torch.allclose(ideal_ctrl, torch.tensor([2.0, -1.0], device=device))


def test_ideal_pd_effort_clamping(device, robot_xml):
  """IdealPd clamps computed torques to [-effort_limit, effort_limit]."""
  kp, kv = 100.0, 10.0
  effort_limit = 5.0

  ideal_entity = create_entity_with_actuator(
    robot_xml,
    IdealPdActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=effort_limit,
      stiffness=kp,
      damping=kv,
    ),
  )

  ideal_entity, ideal_sim = initialize_entity(ideal_entity, device)

  joint_pos = torch.tensor([[0.0, 0.0]], device=device)
  joint_vel = torch.tensor([[0.0, 0.0]], device=device)
  ideal_entity.write_joint_state_to_sim(joint_pos, joint_vel)

  ideal_entity.set_joint_position_target(torch.tensor([[1.0, -1.0]], device=device))
  ideal_entity.set_joint_velocity_target(torch.zeros(1, 2, device=device))
  ideal_entity.set_joint_effort_target(torch.zeros(1, 2, device=device))
  ideal_entity.write_data_to_sim()

  ideal_ctrl = ideal_sim.data.ctrl[0]
  assert torch.allclose(
    ideal_ctrl, torch.tensor([effort_limit, -effort_limit], device=device)
  )
