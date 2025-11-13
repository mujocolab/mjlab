"""Tests for actuator module."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  IdealPdActuatorCfg,
)
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

SIMPLE_ROBOT_XML = """
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


def create_entity_with_actuator(actuator_cfg):
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(SIMPLE_ROBOT_XML),
    articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg,)),
  )
  return Entity(cfg)


def initialize_entity(entity, device, num_envs=1):
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def test_builtin_pd_actuator_compute(device):
  """BuiltinPositionActuator writes position targets to ctrl."""
  actuator_cfg = BuiltinPositionActuatorCfg(
    joint_names_expr=("joint.*",), stiffness=50.0, damping=5.0
  )
  entity = create_entity_with_actuator(actuator_cfg)
  entity, sim = initialize_entity(entity, device)

  entity.set_joint_position_target(torch.tensor([[0.5, -0.3]], device=device))
  entity.write_data_to_sim()

  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([0.5, -0.3], device=device))


def test_ideal_pd_actuator_compute(device):
  """IdealPdActuator computes torques via explicit PD control."""
  actuator_cfg = IdealPdActuatorCfg(
    joint_names_expr=("joint.*",), effort_limit=100.0, stiffness=50.0, damping=5.0
  )
  entity = create_entity_with_actuator(actuator_cfg)
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


def test_targets_cleared_on_reset(device):
  """Entity.reset() zeros all targets."""
  actuator_cfg = BuiltinPositionActuatorCfg(
    joint_names_expr=("joint.*",), stiffness=50.0, damping=5.0
  )
  entity = create_entity_with_actuator(actuator_cfg)
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
