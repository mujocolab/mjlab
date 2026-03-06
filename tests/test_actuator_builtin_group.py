"""Tests for BuiltinActuatorGroup."""

import mujoco
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator import (
  BuiltinMotorActuatorCfg,
  BuiltinPositionActuatorCfg,
  IdealPdActuatorCfg,
)
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

ROBOT_XML = load_fixture_xml("floating_base_articulated")

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


def create_entity(actuator_cfgs, robot_xml=ROBOT_XML, **kwargs):
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(actuators=actuator_cfgs),
    **kwargs,
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
    target_names_expr=("joint.*",), stiffness=50.0, damping=5.0
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
    target_names_expr=("joint.*",), effort_limit=100.0
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
    target_names_expr=("joint1",), stiffness=50.0, damping=5.0
  )
  motor_cfg = BuiltinMotorActuatorCfg(target_names_expr=("joint2",), effort_limit=100.0)
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
    target_names_expr=("joint1",), stiffness=50.0, damping=5.0
  )
  custom_cfg = IdealPdActuatorCfg(
    target_names_expr=("joint2",), effort_limit=100.0, stiffness=50.0, damping=5.0
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
  """Controls are written correctly when actuator order differs from joint order."""
  # Actuators: position on joint2, motor on joint1+joint3.
  # MuJoCo ctrl order follows declaration: [joint2, joint1, joint3].
  position_cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint2",), stiffness=50.0, damping=5.0
  )
  motor_cfg = BuiltinMotorActuatorCfg(
    target_names_expr=("joint1", "joint3"), effort_limit=100.0
  )
  entity = create_entity((position_cfg, motor_cfg), robot_xml=ROBOT_XML_3JOINT)
  entity, sim = initialize_entity(entity, device, num_envs=1)

  entity.set_joint_position_target(torch.tensor([[10.0, 20.0, 30.0]], device=device))
  entity.set_joint_effort_target(torch.tensor([[100.0, 200.0, 300.0]], device=device))
  entity.write_data_to_sim()

  # ctrl follows declaration order: joint2=20, joint1=100, joint3=300.
  expected = torch.tensor([20.0, 100.0, 300.0], device=device)
  assert torch.allclose(sim.data.ctrl[0], expected)


def test_sort_actuators(device):
  """sort_actuators=True reorders ctrl to match joint definition order."""
  position_cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint2",), stiffness=50.0, damping=5.0
  )
  motor_cfg = BuiltinMotorActuatorCfg(
    target_names_expr=("joint1", "joint3"), effort_limit=100.0
  )
  entity = create_entity(
    (position_cfg, motor_cfg),
    robot_xml=ROBOT_XML_3JOINT,
    sort_actuators=True,
  )
  entity, sim = initialize_entity(entity, device, num_envs=1)

  entity.set_joint_position_target(torch.tensor([[10.0, 20.0, 30.0]], device=device))
  entity.set_joint_effort_target(torch.tensor([[100.0, 200.0, 300.0]], device=device))
  entity.write_data_to_sim()

  # Sorted by joint order: joint1=100, joint2=20, joint3=300.
  expected = torch.tensor([100.0, 20.0, 300.0], device=device)
  assert torch.allclose(sim.data.ctrl[0], expected)


def test_position_actuator_ctrlrange_matches_forcerange(device):
  """Commanding to ctrlrange limits produces forces matching forcerange."""
  actuator_cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint.*",),
    stiffness=50.0,
    damping=5.0,
    effort_limit=100.0,
  )
  entity = create_entity((actuator_cfg,), robot_xml=ROBOT_XML_3JOINT)
  model = entity.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  n = model.nu
  ctrlrange = torch.tensor(model.actuator_ctrlrange, device=device, dtype=torch.float32)
  forcerange = torch.tensor(
    model.actuator_forcerange, device=device, dtype=torch.float32
  )
  zeros = torch.zeros((1, n), device=device)

  for col, label in [(1, "max"), (0, "min")]:
    entity.write_joint_state_to_sim(position=zeros, velocity=zeros)
    entity.set_joint_position_target(ctrlrange[:, col].unsqueeze(0))
    entity.set_joint_effort_target(zeros)
    entity.write_data_to_sim()
    sim.step()

    forces = sim.data.actuator_force[0]
    assert torch.allclose(forces, forcerange[:, col], rtol=0.05), (
      f"Forces at ctrl {label} {forces} not within 5% of "
      f"forcerange {label} {forcerange[:, col]}"
    )
