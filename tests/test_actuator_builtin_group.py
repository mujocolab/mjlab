"""Tests for BuiltinActuatorGroup."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import (
  BuiltinMotorActuatorCfg,
  BuiltinPositionActuatorCfg,
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

ROBOT_XML_3JOINT = """
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
      <body name="link3" pos="0 0 0">
        <joint name="joint3" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link3_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def create_entity(actuator_cfgs, robot_xml=ROBOT_XML):
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(actuators=actuator_cfgs),
  )
  return Entity(cfg)


def initialize_entity(entity, device, num_envs=1):
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def test_position_actuator_batched(device):
  """BuiltinPositionActuator writes position targets via batched path."""
  actuator_cfg = BuiltinPositionActuatorCfg(
    joint_names_expr=("joint.*",), stiffness=50.0, damping=5.0
  )
  entity = create_entity((actuator_cfg,))
  entity, sim = initialize_entity(entity, device)

  entity.set_joint_position_target(torch.tensor([[0.5, -0.3]], device=device))
  entity.write_data_to_sim()

  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([0.5, -0.3], device=device))


def test_motor_actuator_batched(device):
  """BuiltinMotorActuator writes effort targets via batched path."""
  actuator_cfg = BuiltinMotorActuatorCfg(
    joint_names_expr=("joint.*",), effort_limit=100.0
  )
  entity = create_entity((actuator_cfg,))
  entity, sim = initialize_entity(entity, device)

  entity.set_joint_effort_target(torch.tensor([[10.0, -5.0]], device=device))
  entity.write_data_to_sim()

  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([10.0, -5.0], device=device))


def test_mixed_builtin_actuators(device):
  """Multiple builtin actuator types can coexist and all use batched path."""
  position_cfg = BuiltinPositionActuatorCfg(
    joint_names_expr=("joint1",), stiffness=50.0, damping=5.0
  )
  motor_cfg = BuiltinMotorActuatorCfg(joint_names_expr=("joint2",), effort_limit=100.0)
  entity = create_entity((position_cfg, motor_cfg))
  entity, sim = initialize_entity(entity, device)

  entity.set_joint_position_target(torch.tensor([[0.5, 0.0]], device=device))
  entity.set_joint_effort_target(torch.tensor([[0.0, -3.0]], device=device))
  entity.write_data_to_sim()

  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([0.5, -3.0], device=device))


def test_builtin_and_custom_actuators(device):
  """Builtin actuators use batched path, custom actuators use compute()."""
  builtin_cfg = BuiltinPositionActuatorCfg(
    joint_names_expr=("joint1",), stiffness=50.0, damping=5.0
  )
  custom_cfg = IdealPdActuatorCfg(
    joint_names_expr=("joint2",), effort_limit=100.0, stiffness=50.0, damping=5.0
  )
  entity = create_entity((builtin_cfg, custom_cfg))
  entity, sim = initialize_entity(entity, device)

  entity.write_joint_state_to_sim(
    position=torch.tensor([[0.0, 0.0]], device=device),
    velocity=torch.tensor([[0.0, 0.0]], device=device),
  )

  entity.set_joint_position_target(torch.tensor([[0.5, 0.2]], device=device))
  entity.set_joint_velocity_target(torch.tensor([[0.0, 0.0]], device=device))
  entity.set_joint_effort_target(torch.tensor([[0.0, 0.0]], device=device))
  entity.write_data_to_sim()

  ctrl = sim.data.ctrl[0]
  # joint1: builtin position -> ctrl = 0.5
  # joint2: ideal pd -> ctrl = kp * (0.2 - 0.0) = 50.0 * 0.2 = 10.0
  assert torch.allclose(ctrl, torch.tensor([0.5, 10.0], device=device))


def test_builtin_group_mismatched_indices(device):
  """Regression test for ctrl_ids/joint_ids swap bug in BuiltinActuatorGroup.

  When actuators are added in different order than joints, ctrl_ids != joint_ids.
  This test verifies controls are written to the correct indices.
  """
  # Add actuators in different order than joints to ensure ctrl_ids != joint_ids.
  # joint1=0, joint2=1, joint3=2 but position actuator on joint2 (ctrl=0),
  # motor actuator on joint1,joint3 (ctrl=1,2).
  position_cfg = BuiltinPositionActuatorCfg(
    joint_names_expr=("joint2",), stiffness=50.0, damping=5.0
  )
  motor_cfg = BuiltinMotorActuatorCfg(
    joint_names_expr=("joint1", "joint3"), effort_limit=100.0
  )
  entity = create_entity((position_cfg, motor_cfg), robot_xml=ROBOT_XML_3JOINT)
  entity, sim = initialize_entity(entity, device, num_envs=1)

  # Verify test setup: at least one actuator has ctrl_ids != joint_ids.
  has_mismatch = any(
    not torch.equal(act.ctrl_ids, act.joint_ids) for act in entity._actuators
  )
  assert has_mismatch, "Test setup failed: need ctrl_ids != joint_ids"

  # Set targets indexed by joint_id: joint1=10.0, joint2=20.0, joint3=30.0
  entity.set_joint_position_target(torch.tensor([[10.0, 20.0, 30.0]], device=device))
  entity.set_joint_effort_target(torch.tensor([[100.0, 200.0, 300.0]], device=device))
  entity.write_data_to_sim()

  # Expected ctrl values: position on joint2 -> ctrl[0]=20.0,
  # motor on joint1 -> ctrl[1]=100.0, motor on joint3 -> ctrl[2]=300.0
  assert torch.allclose(
    sim.data.ctrl[0], torch.tensor([20.0, 100.0, 300.0], device=device)
  ), f"Got {sim.data.ctrl[0]}, expected [20.0, 100.0, 300.0]"
