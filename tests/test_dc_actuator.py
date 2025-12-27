"""Tests for DC motor actuator torque-speed curve."""

import pytest
import torch
from conftest import (
  create_entity_with_actuator,
  get_test_device,
  initialize_entity,
  load_fixture_xml,
)

from mjlab.actuator import DcMotorActuatorCfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def robot_xml():
  return load_fixture_xml("floating_base_articulated")


def test_dc_motor_stall_torque(device, robot_xml):
  """DC motor produces full saturation_effort at zero velocity."""
  kp = 100.0
  kd = 10.0
  saturation_effort = 20.0
  velocity_limit = 30.0

  entity = create_entity_with_actuator(
    robot_xml,
    DcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=saturation_effort,  # Set to saturation to not constrain.
      stiffness=kp,
      damping=kd,
      saturation_effort=saturation_effort,
      velocity_limit=velocity_limit,
    ),
  )

  entity, sim = initialize_entity(entity, device)

  # Zero velocity, large position error to produce high PD torque.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[0.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  # Set target to produce positive torque demand >> saturation_effort.
  entity.set_joint_position_target(torch.tensor([[2.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # At zero velocity, should be clipped to saturation_effort.
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([saturation_effort], device=device))


def test_dc_motor_zero_torque_at_max_velocity(device, robot_xml):
  """DC motor produces zero torque at maximum velocity."""
  kp = 100.0
  kd = 0.0
  saturation_effort = 20.0
  velocity_limit = 30.0

  entity = create_entity_with_actuator(
    robot_xml,
    DcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=saturation_effort,  # Set to saturation to not constrain.
      stiffness=kp,
      damping=kd,
      saturation_effort=saturation_effort,
      velocity_limit=velocity_limit,
    ),
  )

  entity, sim = initialize_entity(entity, device)

  # At max velocity, large position error to produce high PD torque.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[velocity_limit]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  # Set target to produce positive torque demand >> saturation_effort.
  entity.set_joint_position_target(torch.tensor([[2.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # At max velocity, should produce zero torque.
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([0.0], device=device), atol=1e-5)


def test_dc_motor_linear_torque_speed_curve(device, robot_xml):
  """DC motor torque varies linearly between zero and max velocity."""
  kp = 100.0
  kd = 0.0
  saturation_effort = 20.0
  velocity_limit = 30.0

  entity = create_entity_with_actuator(
    robot_xml,
    DcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=saturation_effort,  # Set to saturation to not constrain.
      stiffness=kp,
      damping=kd,
      saturation_effort=saturation_effort,
      velocity_limit=velocity_limit,
    ),
  )

  entity, sim = initialize_entity(entity, device)

  # Test at half max velocity: should produce half saturation_effort.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[velocity_limit / 2.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  # Large position error to produce high PD torque.
  entity.set_joint_position_target(torch.tensor([[2.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # At half velocity, should produce half saturation_effort.
  ctrl = sim.data.ctrl[0]
  expected = saturation_effort * 0.5
  assert torch.allclose(ctrl, torch.tensor([expected], device=device), rtol=1e-4)


def test_dc_motor_effort_limit_constrains_output(device, robot_xml):
  """Continuous effort_limit constrains output below saturation_effort."""
  kp = 100.0
  kd = 0.0
  saturation_effort = 20.0
  effort_limit = 5.0
  velocity_limit = 30.0

  entity = create_entity_with_actuator(
    robot_xml,
    DcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=effort_limit,
      stiffness=kp,
      damping=kd,
      saturation_effort=saturation_effort,
      velocity_limit=velocity_limit,
    ),
  )

  entity, sim = initialize_entity(entity, device)

  # Zero velocity: would produce saturation_effort without effort_limit.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[0.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  # Large position error to produce high PD torque.
  entity.set_joint_position_target(torch.tensor([[2.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # Should be clamped to effort_limit, not saturation_effort.
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([effort_limit], device=device))


def test_dc_motor_negative_velocity_behavior(device, robot_xml):
  """DC motor handles negative velocities symmetrically."""
  kp = 100.0
  kd = 0.0
  saturation_effort = 20.0
  velocity_limit = 30.0

  entity = create_entity_with_actuator(
    robot_xml,
    DcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=saturation_effort,  # Set to saturation to not constrain
      stiffness=kp,
      damping=kd,
      saturation_effort=saturation_effort,
      velocity_limit=velocity_limit,
    ),
  )

  entity, sim = initialize_entity(entity, device)

  # Negative velocity, target produces negative torque demand.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[-velocity_limit / 2.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  # Set target to produce negative torque demand.
  entity.set_joint_position_target(torch.tensor([[-2.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # Should produce -half saturation_effort.
  ctrl = sim.data.ctrl[0]
  expected = -saturation_effort * 0.5
  assert torch.allclose(ctrl, torch.tensor([expected], device=device), rtol=1e-4)


def test_dc_motor_corner_velocity_transition(device, robot_xml):
  """DC motor transitions correctly at corner velocity where curves intersect."""
  kp = 100.0
  kd = 0.0
  saturation_effort = 20.0
  effort_limit = 10.0
  velocity_limit = 30.0

  entity = create_entity_with_actuator(
    robot_xml,
    DcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=effort_limit,
      stiffness=kp,
      damping=kd,
      saturation_effort=saturation_effort,
      velocity_limit=velocity_limit,
    ),
  )

  entity, sim = initialize_entity(entity, device)

  # Corner velocity: where torque-speed curve intersects effort_limit.
  # vel_corner = velocity_limit * (1 - effort_limit / saturation_effort)
  vel_corner = velocity_limit * (1.0 - effort_limit / saturation_effort)

  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[vel_corner]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  # Large position error to produce high PD torque.
  entity.set_joint_position_target(torch.tensor([[2.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # At corner velocity, should produce exactly effort_limit.
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([effort_limit], device=device), rtol=1e-4)


def test_dc_motor_warns_when_effort_limit_is_inf():
  """DcMotorActuatorCfg warns when effort_limit is inf."""
  import warnings

  # inf triggers both warnings (is inf + exceeds saturation), so catch both.
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    DcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      stiffness=100.0,
      damping=10.0,
      saturation_effort=20.0,
      velocity_limit=30.0,
      # effort_limit intentionally not set (defaults to inf).
    )
    # Should trigger the "is inf" warning.
    assert len(w) >= 1
    assert any("effort_limit is set to inf" in str(warning.message) for warning in w)


def test_dc_motor_warns_when_effort_limit_exceeds_saturation():
  """DcMotorActuatorCfg warns when effort_limit > saturation_effort."""
  with pytest.warns(UserWarning, match="effort_limit.*exceeds saturation_effort"):
    DcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      stiffness=100.0,
      damping=10.0,
      saturation_effort=20.0,
      velocity_limit=30.0,
      effort_limit=25.0,  # > saturation_effort.
    )
