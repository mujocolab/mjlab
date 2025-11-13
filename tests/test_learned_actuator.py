"""Tests for learned MLP actuator."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import LearnedMlpActuator, LearnedMlpActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

ROBOT_XML_TWO_JOINTS = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
          <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def identity_network_file(tmp_path_factory):
  """Create a simple identity network: output = sum of all inputs."""

  class IdentityNet(torch.nn.Module):
    def forward(self, x):
      # Sum all inputs and return as single output per joint.
      return x.sum(dim=1, keepdim=True)

  net = IdentityNet()
  tmp_dir = tmp_path_factory.mktemp("networks")
  network_path = tmp_dir / "identity_net.pt"
  torch.jit.script(net).save(str(network_path))
  return str(network_path)


@pytest.fixture(scope="module")
def constant_network_file(tmp_path_factory):
  """Create a network that returns constant value regardless of input."""

  class ConstantNet(torch.nn.Module):
    def forward(self, x):
      # Return constant 10.0 for each joint.
      return torch.full((x.shape[0], 1), 10.0, device=x.device)

  net = ConstantNet()
  tmp_dir = tmp_path_factory.mktemp("networks")
  network_path = tmp_dir / "constant_net.pt"
  torch.jit.script(net).save(str(network_path))
  return str(network_path)


@pytest.fixture(scope="module")
def subtract_network_file(tmp_path_factory):
  """Create a network that subtracts second half from first half of inputs.

  For input [a, b, c, d], returns (a + b) - (c + d).
  This is sensitive to input order, unlike sum-based networks.
  """

  class SubtractNet(torch.nn.Module):
    def forward(self, x):
      # Split input in half and subtract second half from first half.
      mid = x.shape[1] // 2
      first_half = x[:, :mid].sum(dim=1, keepdim=True)
      second_half = x[:, mid:].sum(dim=1, keepdim=True)
      return first_half - second_half

  net = SubtractNet()
  tmp_dir = tmp_path_factory.mktemp("networks")
  network_path = tmp_dir / "subtract_net.pt"
  torch.jit.script(net).save(str(network_path))
  return str(network_path)


def create_entity_with_actuator(actuator_cfg):
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg,)),
  )
  return Entity(cfg)


def initialize_entity(entity, device, num_envs=1):
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def test_learned_mlp_network_loads(device, identity_network_file):
  """Verify network loads from TorchScript file and initializes."""
  entity = create_entity_with_actuator(
    LearnedMlpActuatorCfg(
      joint_names_expr=("joint.*",),
      network_file=identity_network_file,
      pos_scale=1.0,
      vel_scale=1.0,
      torque_scale=1.0,
      input_order="pos_vel",
      history_length=1,
      saturation_effort=100.0,
      velocity_limit=30.0,
      effort_limit=50.0,
    )
  )

  entity, sim = initialize_entity(entity, device)

  # Verify network was loaded.
  actuator = entity.actuators[0]
  assert isinstance(actuator, LearnedMlpActuator)
  assert actuator.network is not None
  assert actuator._pos_error_history is not None
  assert actuator._vel_history is not None


def test_learned_mlp_history_indexing(device, identity_network_file):
  """Test that history_length uses consecutive timesteps."""
  entity = create_entity_with_actuator(
    LearnedMlpActuatorCfg(
      joint_names_expr=("joint.*",),
      network_file=identity_network_file,
      pos_scale=1.0,
      vel_scale=0.0,  # Zero out velocity contribution.
      torque_scale=1.0,
      input_order="pos_vel",
      history_length=3,  # Use current + 2 previous timesteps.
      saturation_effort=100.0,
      velocity_limit=30.0,
      effort_limit=100.0,
    )
  )

  entity, sim = initialize_entity(entity, device)

  # Set up state at rest.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[0.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  # Set position targets to create known position errors.
  # Step 0: pos_error = 1.0
  # Step 1: pos_error = 2.0
  # Step 2: pos_error = 3.0
  targets = [
    torch.tensor([[1.0]], device=device),
    torch.tensor([[2.0]], device=device),
    torch.tensor([[3.0]], device=device),
  ]

  for target in targets:
    entity.set_joint_position_target(target)
    entity.set_joint_velocity_target(joint_vel)
    entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
    entity.write_data_to_sim()

  # After 3 steps with history_length=3:
  # Network input: [current, -1 step, -2 steps] = [3.0, 2.0, 1.0]
  # Identity network sums: 3.0 + 2.0 + 1.0 = 6.0
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([6.0], device=device), atol=1e-4)


def test_learned_mlp_input_order_pos_vel(device, subtract_network_file):
  """Verify input_order='pos_vel' concatenates position then velocity."""
  entity = create_entity_with_actuator(
    LearnedMlpActuatorCfg(
      joint_names_expr=("joint.*",),
      network_file=subtract_network_file,
      pos_scale=1.0,
      vel_scale=1.0,
      torque_scale=1.0,
      input_order="pos_vel",
      history_length=1,
      saturation_effort=100.0,
      velocity_limit=30.0,
      effort_limit=100.0,
    )
  )

  entity, sim = initialize_entity(entity, device)

  # Set state: pos_error = 2.0, vel = 3.0.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[3.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  entity.set_joint_position_target(torch.tensor([[2.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # Network input: [pos_error, vel] = [2.0, 3.0]
  # Subtract network: 2.0 - 3.0 = -1.0
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([-1.0], device=device), atol=1e-4)


def test_learned_mlp_input_order_vel_pos(device, subtract_network_file):
  """Verify input_order='vel_pos' concatenates velocity then position."""
  entity = create_entity_with_actuator(
    LearnedMlpActuatorCfg(
      joint_names_expr=("joint.*",),
      network_file=subtract_network_file,
      pos_scale=1.0,
      vel_scale=1.0,
      torque_scale=1.0,
      input_order="vel_pos",
      history_length=1,
      saturation_effort=100.0,
      velocity_limit=30.0,
      effort_limit=100.0,
    )
  )

  entity, sim = initialize_entity(entity, device)

  # Set state: pos_error = 2.0, vel = 3.0.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[3.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  entity.set_joint_position_target(torch.tensor([[2.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # Network input: [vel, pos_error] = [3.0, 2.0]
  # Subtract network: 3.0 - 2.0 = 1.0 (opposite of pos_vel!)
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([1.0], device=device), atol=1e-4)


def test_learned_mlp_scaling_applied(device, identity_network_file):
  """Test pos_scale, vel_scale, torque_scale are applied correctly."""
  entity = create_entity_with_actuator(
    LearnedMlpActuatorCfg(
      joint_names_expr=("joint.*",),
      network_file=identity_network_file,
      pos_scale=2.0,
      vel_scale=3.0,
      torque_scale=4.0,
      input_order="pos_vel",
      history_length=1,
      saturation_effort=100.0,
      velocity_limit=30.0,
      effort_limit=100.0,
    )
  )

  entity, sim = initialize_entity(entity, device)

  # Set state: pos_error = 1.0, vel = 1.0.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[1.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  entity.set_joint_position_target(torch.tensor([[1.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # Network input: [1.0 * 2.0, 1.0 * 3.0] = [2.0, 3.0]
  # Identity network sums: 2.0 + 3.0 = 5.0
  # Output scaled: 5.0 * 4.0 = 20.0
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([20.0], device=device), atol=1e-4)


def test_learned_mlp_reset_clears_history(device, identity_network_file):
  """Test reset zeroes history buffers for specified environments."""
  entity = create_entity_with_actuator(
    LearnedMlpActuatorCfg(
      joint_names_expr=("joint.*",),
      network_file=identity_network_file,
      pos_scale=1.0,
      vel_scale=1.0,
      torque_scale=1.0,
      input_order="pos_vel",
      history_length=2,
      saturation_effort=100.0,
      velocity_limit=30.0,
      effort_limit=100.0,
    )
  )

  entity, sim = initialize_entity(entity, device, num_envs=2)

  # Set state and targets to fill buffers.
  joint_pos = torch.tensor([[0.0], [0.0]], device=device)
  joint_vel = torch.tensor([[1.0], [1.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  entity.set_joint_position_target(torch.tensor([[5.0], [5.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(2, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(2, 1, device=device))
  entity.write_data_to_sim()

  # Reset only env 0.
  entity.reset(torch.tensor([0], device=device))

  # Check history buffers.
  actuator = entity.actuators[0]
  assert isinstance(actuator, LearnedMlpActuator)
  assert actuator._pos_error_history is not None
  assert actuator._vel_history is not None
  assert actuator._pos_error_history.current_length[0] == 0
  assert actuator._pos_error_history.current_length[1] > 0
  assert actuator._vel_history.current_length[0] == 0
  assert actuator._vel_history.current_length[1] > 0


def test_learned_mlp_inherits_dc_motor_limits(device, constant_network_file):
  """Test that DC motor saturation_effort and velocity limits apply."""
  saturation_effort = 20.0
  velocity_limit = 30.0

  entity = create_entity_with_actuator(
    LearnedMlpActuatorCfg(
      joint_names_expr=("joint.*",),
      network_file=constant_network_file,  # Always outputs 10.0.
      pos_scale=1.0,
      vel_scale=1.0,
      torque_scale=100.0,  # Scale to 1000.0 to exceed limits.
      input_order="pos_vel",
      history_length=1,
      saturation_effort=saturation_effort,
      velocity_limit=velocity_limit,
      effort_limit=float("inf"),
    )
  )

  entity, sim = initialize_entity(entity, device)

  # At zero velocity, should be clipped to saturation_effort.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[0.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  entity.set_joint_position_target(torch.tensor([[1.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # Network outputs 10.0, scaled by 100.0 = 1000.0.
  # Should be clipped to saturation_effort = 20.0.
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([saturation_effort], device=device))


def test_learned_mlp_dc_motor_zero_torque_at_max_velocity(
  device, constant_network_file
):
  """Test that DC motor produces zero torque at max velocity."""
  saturation_effort = 20.0
  velocity_limit = 30.0

  entity = create_entity_with_actuator(
    LearnedMlpActuatorCfg(
      joint_names_expr=("joint.*",),
      network_file=constant_network_file,  # Always outputs 10.0.
      pos_scale=1.0,
      vel_scale=1.0,
      torque_scale=100.0,  # Scale to 1000.0.
      input_order="pos_vel",
      history_length=1,
      saturation_effort=saturation_effort,
      velocity_limit=velocity_limit,
      effort_limit=float("inf"),
    )
  )

  entity, sim = initialize_entity(entity, device)

  # At max velocity, should produce zero torque.
  joint_pos = torch.tensor([[0.0]], device=device)
  joint_vel = torch.tensor([[velocity_limit]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  entity.set_joint_position_target(torch.tensor([[1.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))
  entity.write_data_to_sim()

  # At max velocity, DC motor should clip to zero.
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([0.0], device=device), atol=1e-5)


def test_learned_mlp_multi_joint_reshaping(device, identity_network_file):
  """Test that multi-joint reshaping works correctly."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_TWO_JOINTS),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        LearnedMlpActuatorCfg(
          joint_names_expr=("joint.*",),
          network_file=identity_network_file,
          pos_scale=1.0,
          vel_scale=1.0,
          torque_scale=1.0,
          input_order="pos_vel",
          history_length=1,
          saturation_effort=100.0,
          velocity_limit=30.0,
          effort_limit=100.0,
        ),
      )
    ),
  )

  entity = Entity(cfg)
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  # Set different states for each joint.
  # Joint 1: pos_error = 1.0, vel = 2.0
  # Joint 2: pos_error = 3.0, vel = 4.0
  joint_pos = torch.tensor([[0.0, 0.0]], device=device)
  joint_vel = torch.tensor([[2.0, 4.0]], device=device)
  entity.write_joint_state_to_sim(joint_pos, joint_vel)

  entity.set_joint_position_target(torch.tensor([[1.0, 3.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 2, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 2, device=device))
  entity.write_data_to_sim()

  # Network input for joint 1: [1.0, 2.0] → sum = 3.0
  # Network input for joint 2: [3.0, 4.0] → sum = 7.0
  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([3.0, 7.0], device=device), atol=1e-4)
