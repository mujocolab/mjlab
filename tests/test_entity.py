"""Tests for entity module."""

import mujoco
import pytest
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.entity.config import EntityArticulationInfoCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.spec_editor.spec_editor_config import ActuatorCfg


def get_test_device() -> str:
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def fixed_base_xml():
  """XML for a simple fixed-base entity."""
  return """
    <mujoco>
      <worldbody>
        <body name="object" pos="0 0 0.5">
          <geom name="object_geom" type="box" size="0.1 0.1 0.1" rgba="0.8 0.3 0.3 1"/>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def floating_base_xml():
  """XML for a floating-base entity with freejoint."""
  return """
    <mujoco>
      <worldbody>
        <body name="object" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="object_geom" type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1" mass="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def articulated_xml():
  """XML for an articulated entity with joints."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
          <body name="link1" pos="0 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
            <site name="site1" pos="0 0 0"/>
          </body>
          <body name="link2" pos="0 0 0">
            <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
          </body>
        </body>
      </worldbody>
      <sensor>
        <jointpos name="joint1_pos" joint="joint1"/>
      </sensor>
    </mujoco>
    """


@pytest.fixture
def fixed_articulated_xml():
  """XML for a fixed-base articulated entity (e.g., robot arm bolted to ground)."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 0.5">
          <geom name="base_geom" type="cylinder" size="0.1 0.05" mass="5.0"/>
          <body name="link1" pos="0 0 0.1">
            <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom name="link1_geom" type="box" size="0.05 0.05 0.2" mass="1.0"/>
            <body name="link2" pos="0 0 0.4">
              <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
              <geom name="link2_geom" type="box" size="0.05 0.05 0.15" mass="0.5"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def fixed_articulated_entity(fixed_articulated_xml, actuator_cfg):
  """Create a fixed-base articulated entity."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(fixed_articulated_xml),
    articulation=actuator_cfg,
  )
  return Entity(cfg)


@pytest.fixture
def actuator_cfg():
  """Standard actuator configuration."""
  return EntityArticulationInfoCfg(
    actuators=(
      ActuatorCfg(
        joint_names_expr=["joint1", "joint2"],
        effort_limit=1.0,
        stiffness=1.0,
        damping=1.0,
      ),
    )
  )


@pytest.fixture
def fixed_base_entity(fixed_base_xml):
  """Create a fixed-base entity."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(fixed_base_xml))
  return Entity(cfg)


@pytest.fixture
def floating_base_entity(floating_base_xml):
  """Create a floating-base entity."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(floating_base_xml))
  return Entity(cfg)


@pytest.fixture
def articulated_entity(articulated_xml, actuator_cfg):
  """Create an articulated entity with actuators."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(articulated_xml),
    articulation=actuator_cfg,
  )
  return Entity(cfg)


@pytest.fixture
def initialized_floating_entity(floating_base_entity, device):
  """Create an initialized floating-base entity with simulation."""

  entity = floating_base_entity
  model = entity.compile()

  sim_cfg = SimulationCfg(device=device)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model)

  entity.initialize(model, sim.model, sim.data, device)

  return entity, sim


@pytest.fixture
def initialized_articulated_entity(articulated_entity, device):
  """Create an initialized articulated entity with simulation."""

  entity = articulated_entity
  model = entity.compile()

  sim_cfg = SimulationCfg(device=device)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model)

  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


class TestEntityProperties:
  """Test entity property detection and element counts."""

  @pytest.mark.parametrize(
    "entity_fixture,expected",
    [
      (
        "fixed_base_entity",
        {
          "is_fixed_base": True,
          "is_articulated": False,
          "is_actuated": False,
          "num_bodies": 1,
          "num_joints": 0,
          "num_actuators": 0,
        },
      ),
      (
        "floating_base_entity",
        {
          "is_fixed_base": False,
          "is_articulated": False,
          "is_actuated": False,
          "num_bodies": 1,
          "num_joints": 0,
          "num_actuators": 0,
        },
      ),
      (
        "articulated_entity",
        {
          "is_fixed_base": False,
          "is_articulated": True,
          "is_actuated": True,
          "num_bodies": 3,
          "num_joints": 2,
          "num_actuators": 2,
        },
      ),
      (
        "fixed_articulated_entity",
        {
          "is_fixed_base": True,
          "is_articulated": True,
          "is_actuated": True,
          "num_bodies": 3,
          "num_joints": 2,
          "num_actuators": 2,
        },
      ),
    ],
  )
  def test_entity_properties(self, entity_fixture, expected, request):
    """Test entity type properties and element counts."""
    entity = request.getfixturevalue(entity_fixture)

    for prop, value in expected.items():
      assert getattr(entity, prop) == value


class TestFindMethods:
  """Test entity element finding methods."""

  def test_find_methods(self, articulated_entity):
    """Test find methods with exact and regex matches."""
    # Test exact matches.
    assert articulated_entity.find_bodies("base")[1] == ["base"]
    assert articulated_entity.find_joints("joint1")[1] == ["joint1"]
    assert articulated_entity.find_sites("site1")[1] == ["site1"]

    # Test regex matches.
    assert articulated_entity.find_bodies("link.*")[1] == ["link1", "link2"]
    assert articulated_entity.find_joints("joint.*")[1] == ["joint1", "joint2"]

    # Test subset filtering.
    _, names = articulated_entity.find_joints(
      "joint1", joint_subset=["joint1", "joint2"]
    )
    assert names == ["joint1"]

    # Test error on invalid subset.
    with pytest.raises(ValueError, match="Not all regular expressions are matched"):
      articulated_entity.find_joints("joint1", joint_subset=["joint2"])


class TestStateManagement:
  """Test reading and writing entity states."""

  def test_root_state_floating_base(self, initialized_floating_entity, device):
    """Test root state operations affect simulation correctly."""
    entity, sim = initialized_floating_entity

    # Set entity with specific state.
    # fmt: off
    root_state = torch.tensor([
        1.0, 2.0, 3.0,           # position
        1.0, 0.0, 0.0, 0.0,      # quaternion (identity)
        0.5, 0.0, 0.0,           # linear velocity in X
        0.0, 0.0, 0.2            # angular velocity around Z
    ], device=device).unsqueeze(0)
    # fmt: on

    entity.write_root_state_to_sim(root_state)

    # Verify the state was actually written.
    q_slice = entity.data.indexing.free_joint_q_adr
    v_slice = entity.data.indexing.free_joint_v_adr

    assert torch.allclose(sim.data.qpos[:, q_slice], root_state[:, :7])
    assert torch.allclose(sim.data.qvel[:, v_slice], root_state[:, 7:])

    # Step once and verify physics is working (gravity should affect Z velocity).
    initial_z_vel = sim.data.qvel[0, v_slice[2]].item()
    sim.step()
    final_z_vel = sim.data.qvel[0, v_slice[2]].item()

    # Z velocity should decrease (become more negative) due to gravity.
    assert final_z_vel < initial_z_vel, "Gravity should affect Z velocity"

  def test_joint_state_articulated(self, initialized_articulated_entity, device):
    """Test joint state operations and control parameters."""
    entity, sim = initialized_articulated_entity

    # Set joints to specific positions.
    joint_pos = torch.tensor([[0.5, -0.3]], device=device)
    joint_vel = torch.tensor([[0.0, 0.0]], device=device)  # Start with zero velocity.

    entity.write_joint_state_to_sim(joint_pos, joint_vel)

    # Verify state was written
    q_slice = entity.data.indexing.joint_q_adr
    v_slice = entity.data.indexing.joint_v_adr
    assert torch.allclose(sim.data.qpos[:, q_slice], joint_pos)
    assert torch.allclose(sim.data.qvel[:, v_slice], joint_vel)

    # Test stiffness - high stiffness should resist deviation from current position.
    entity.write_joint_stiffness_to_sim(100.0)
    entity.write_joint_damping_to_sim(10.0)

    # Apply force to the system.
    body_ids = entity.find_bodies("link1")[0]
    entity.set_external_force_and_torque(
      forces=torch.tensor([[10.0, 0.0, 0.0]], device=device),
      torques=torch.zeros((1, 3), device=device),
      body_ids=body_ids,
    )

    initial_pos = sim.data.qpos[:, q_slice].clone()

    for _ in range(10):
      sim.step()

    final_pos = sim.data.qpos[:, q_slice]

    # With high stiffness, joints shouldn't deviate much despite force.
    deviation = torch.abs(final_pos - initial_pos).max()
    assert deviation < 1.0, f"High stiffness should limit deviation, got {deviation}"

    # Now test with low stiffness - should deviate more.
    entity.write_joint_stiffness_to_sim(0.1)
    entity.write_joint_damping_to_sim(0.01)

    # Reset to same initial position.
    entity.write_joint_state_to_sim(joint_pos, joint_vel)

    # Apply same force.
    entity.set_external_force_and_torque(
      forces=torch.tensor([[10.0, 0.0, 0.0]], device=device),
      torques=torch.zeros((1, 3), device=device),
      body_ids=body_ids,
    )

    for _ in range(10):
      sim.step()

    final_pos_low_stiff = sim.data.qpos[:, q_slice]
    deviation_low = torch.abs(final_pos_low_stiff - joint_pos).max()

    # With low stiffness, deviation should be larger.
    assert deviation_low > deviation, "Lower stiffness should allow more deviation"

  def test_fixed_base_restrictions(self, fixed_base_entity):
    """Test that fixed-base entities reject root state operations."""
    dummy_state = torch.zeros(1, 13)

    with pytest.raises(ValueError, match="Cannot write root state"):
      fixed_base_entity.write_root_state_to_sim(dummy_state)


class TestExternalForces:
  """Test external force and torque application."""

  def test_force_and_torque_basic(self, initialized_floating_entity):
    """Test forces translate, torques rotate, and forces can be cleared."""
    entity, sim = initialized_floating_entity

    # Apply force in X, torque around Z.
    entity.set_external_force_and_torque(
      forces=torch.tensor([[5.0, 0.0, 0.0]], device=sim.device),
      torques=torch.tensor([[0.0, 0.0, 3.0]], device=sim.device),
    )

    initial_pos = sim.data.qpos[0, :3].clone()
    initial_quat = sim.data.qpos[0, 3:7].clone()

    for _ in range(10):
      sim.step()

    # Verify X translation and rotation occurred.
    assert sim.data.qpos[0, 0] > initial_pos[0], "Force should cause X translation"
    assert not torch.allclose(sim.data.qpos[0, 3:7], initial_quat), (
      "Torque should cause rotation"
    )

    # Verify angular velocity is primarily around Z (relative comparison).
    angular_vel = sim.data.qvel[0, 3:6]
    z_rotation = abs(angular_vel[2])
    xy_rotation = abs(angular_vel[0]) + abs(angular_vel[1])
    assert z_rotation > xy_rotation * 5, "Rotation should be primarily around Z axis"

    # Test force clearing.
    entity.set_external_force_and_torque(
      forces=torch.zeros((1, 3), device=sim.device),
      torques=torch.zeros((1, 3), device=sim.device),
    )

    # Verify forces are cleared.
    body_id = entity.indexing.body_ids[0]
    assert torch.allclose(
      sim.data.xfrc_applied[:, body_id, :], torch.zeros(6, device=sim.device)
    )

    # Verify gravity still works after clearing.
    initial_z = sim.data.qpos[0, 2].clone()
    sim.step()
    assert sim.data.qpos[0, 2] < initial_z, "Should fall due to gravity"

  def test_force_on_specific_body(self, initialized_articulated_entity):
    """Test applying force to specific body in articulated system."""
    entity, sim = initialized_articulated_entity

    # Apply force only to link1.
    body_ids = entity.find_bodies("link1")[0]
    entity.set_external_force_and_torque(
      forces=torch.tensor([[3.0, 0.0, 0.0]], device=sim.device),
      torques=torch.zeros((1, 3), device=sim.device),
      body_ids=body_ids,
    )

    # Verify force applied only to link1.
    link1_id = sim.mj_model.body("link1").id
    base_id = sim.mj_model.body("base").id
    assert torch.allclose(
      sim.data.xfrc_applied[0, link1_id, :3],
      torch.tensor([3.0, 0.0, 0.0], device=sim.device),
    )
    assert torch.allclose(
      sim.data.xfrc_applied[0, base_id, :3], torch.zeros(3, device=sim.device)
    )

    # Verify motion occurs.
    initial_pos = sim.data.xpos[0, link1_id, :].clone()
    for _ in range(10):
      sim.step()
    assert not torch.allclose(sim.data.xpos[0, link1_id, :], initial_pos)

  def test_large_force_stability(self, initialized_floating_entity):
    """Test system handles large forces without numerical issues."""
    entity, sim = initialized_floating_entity

    entity.set_external_force_and_torque(
      forces=torch.tensor([[1e6, 0.0, 0.0]], device=sim.device),
      torques=torch.zeros((1, 3), device=sim.device),
    )

    sim.step()
    assert not torch.any(torch.isnan(sim.data.qpos)), "Should not produce NaN"


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
