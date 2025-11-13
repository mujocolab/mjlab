"""Tests for delayed actuators."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  DelayedActuator,
  DelayedActuatorCfg,
  IdealPdActuatorCfg,
)
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
      <body name="link2" pos="0 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def create_entity_with_delayed_builtin(delay_min_lag=0, delay_max_lag=3):
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        DelayedActuatorCfg(
          joint_names_expr=("joint.*",),
          base_cfg=BuiltinPositionActuatorCfg(
            joint_names_expr=("joint.*",),
            effort_limit=100.0,
            stiffness=80.0,
            damping=10.0,
          ),
          delay_target="position",
          delay_min_lag=delay_min_lag,
          delay_max_lag=delay_max_lag,
        ),
      )
    ),
  )
  return Entity(cfg)


def create_entity_with_delayed_ideal(delay_min_lag=0, delay_max_lag=3):
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        DelayedActuatorCfg(
          joint_names_expr=("joint.*",),
          base_cfg=IdealPdActuatorCfg(
            joint_names_expr=("joint.*",),
            effort_limit=100.0,
            stiffness=80.0,
            damping=10.0,
          ),
          delay_target="position",
          delay_min_lag=delay_min_lag,
          delay_max_lag=delay_max_lag,
        ),
      )
    ),
  )
  return Entity(cfg)


def initialize_entity(entity, device, num_envs=1):
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def test_delayed_builtin_applies_constant_delay(device):
  """Test that delayed builtin actuator delays position targets."""
  entity = create_entity_with_delayed_builtin(delay_min_lag=2, delay_max_lag=2)
  entity, sim = initialize_entity(entity, device)

  # Set position targets for 3 steps.
  targets = [
    torch.tensor([[0.1, 0.2]], device=device),
    torch.tensor([[0.3, 0.4]], device=device),
    torch.tensor([[0.5, 0.6]], device=device),
  ]

  joint_vel = torch.zeros(1, 2, device=device)

  for target in targets:
    entity.set_joint_position_target(target)
    entity.set_joint_velocity_target(joint_vel)
    entity.set_joint_effort_target(torch.zeros(1, 2, device=device))
    entity.write_data_to_sim()

  # After 3 steps with lag=2, the output should be the target from step 0.
  ctrl = sim.data.ctrl[0]
  # With constant lag=2, after 3 appends, we expect target from step 0.
  assert torch.allclose(ctrl, targets[0][0], atol=1e-5)


def test_delayed_ideal_applies_delay(device):
  """Test that delayed ideal actuator delays position targets."""
  entity = create_entity_with_delayed_ideal(delay_min_lag=2, delay_max_lag=2)
  entity, sim = initialize_entity(entity, device)

  joint_pos = torch.zeros(1, 2, device=device)
  joint_vel = torch.zeros(1, 2, device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  # Set position targets for 3 steps.
  targets = [
    torch.tensor([[0.1, 0.2]], device=device),
    torch.tensor([[0.3, 0.4]], device=device),
    torch.tensor([[0.5, 0.6]], device=device),
  ]

  for target in targets:
    entity.set_joint_position_target(target)
    entity.set_joint_velocity_target(joint_vel)
    entity.set_joint_effort_target(torch.zeros(1, 2, device=device))
    entity.write_data_to_sim()
    sim.forward()  # Compute actuator forces

  # The computed torque should use the delayed target from step 0.
  joint_v_adr = entity.indexing.joint_v_adr
  qfrc = sim.data.qfrc_actuator[0, joint_v_adr]

  # Expected torque: kp * (delayed_target - joint_pos) + kd * (0 - joint_vel) + 0
  # = 80.0 * targets[0] + 0 = 80.0 * [0.1, 0.2]
  expected_torque = 80.0 * targets[0][0]
  assert torch.allclose(qfrc, expected_torque, atol=1e-4)


def test_delayed_actuator_reset(device):
  """Test that reset clears the delay buffer."""
  entity = create_entity_with_delayed_builtin(delay_min_lag=1, delay_max_lag=3)
  entity, _ = initialize_entity(entity, device, num_envs=2)

  # Set some targets to fill the buffer.
  entity.set_joint_position_target(torch.ones(2, 2, device=device) * 0.5)
  entity.set_joint_velocity_target(torch.zeros(2, 2, device=device))
  entity.set_joint_effort_target(torch.zeros(2, 2, device=device))
  entity.write_data_to_sim()

  # Reset env 0.
  entity.reset(torch.tensor([0], device=device))

  # Check that delay buffer was reset for env 0.
  actuator = entity.actuators[0]
  assert isinstance(actuator, DelayedActuator)
  assert len(actuator._delay_buffers) > 0
  delay_buffer = next(iter(actuator._delay_buffers.values()))
  assert delay_buffer.current_lags[0] == 0


def test_delayed_actuator_multi_target(device):
  """Test that multiple targets can be delayed simultaneously."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        DelayedActuatorCfg(
          joint_names_expr=("joint.*",),
          base_cfg=IdealPdActuatorCfg(
            joint_names_expr=("joint.*",),
            effort_limit=100.0,
            stiffness=80.0,
            damping=10.0,
          ),
          delay_target=("position", "velocity", "effort"),
          delay_min_lag=2,
          delay_max_lag=2,
        ),
      )
    ),
  )

  entity = Entity(cfg)
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  actuator = entity.actuators[0]
  assert isinstance(actuator, DelayedActuator)
  # Should have 3 delay buffers (one for each target).
  assert len(actuator._delay_buffers) == 3
  assert "position" in actuator._delay_buffers
  assert "velocity" in actuator._delay_buffers
  assert "effort" in actuator._delay_buffers

  # Initialize joints at zero.
  joint_pos = torch.zeros(1, 2, device=device)
  joint_vel = torch.zeros(1, 2, device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  # Set different targets over 3 steps.
  targets = [
    (
      torch.tensor([[0.1, 0.2]], device=device),
      torch.tensor([[0.01, 0.02]], device=device),
    ),
    (
      torch.tensor([[0.3, 0.4]], device=device),
      torch.tensor([[0.03, 0.04]], device=device),
    ),
    (
      torch.tensor([[0.5, 0.6]], device=device),
      torch.tensor([[0.05, 0.06]], device=device),
    ),
  ]

  for pos_target, vel_target in targets:
    entity.set_joint_position_target(pos_target)
    entity.set_joint_velocity_target(vel_target)
    entity.set_joint_effort_target(torch.zeros(1, 2, device=device))
    entity.write_data_to_sim()

  # After 3 steps with lag=2, the delayed targets should be from step 0.
  # Position: [0.1, 0.2], Velocity: [0.01, 0.02]
  # Expected torque: Kp*(0.1 - 0) + Kd*(0.01 - 0) = 80*0.1 + 10*0.01 = 8.1.
  ctrl_ids = actuator.ctrl_ids
  ctrl = sim.data.ctrl[0, ctrl_ids]
  # [80*0.1 + 10*0.01, 80*0.2 + 10*0.02]
  expected = torch.tensor([8.1, 16.2], device=device)
  assert torch.allclose(ctrl, expected, atol=1e-4)
