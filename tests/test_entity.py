"""Tests for entity module."""

import mujoco
import pytest
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.entity.config import EntityArticulationInfoCfg
from mjlab.sim.sim_data import WarpBridge
from mjlab.utils.spec_editor.spec_editor_config import ActuatorCfg

# ============================================================================
# Fixtures
# ============================================================================


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
def initialized_floating_entity(floating_base_entity):
  """Create an initialized floating-base entity with simulation."""
  import mujoco_warp as mjwarp

  entity = floating_base_entity
  model = entity.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetData(model, data)

  wp_model = mjwarp.put_model(model)
  wp_data = mjwarp.put_data(model, data, nworld=1)

  wp_model = WarpBridge(wp_model)
  wp_data = WarpBridge(wp_data)

  entity.initialize(model, wp_model, wp_data, "cuda")
  return entity, wp_data


@pytest.fixture
def initialized_articulated_entity(articulated_entity):
  """Create an initialized articulated entity with simulation."""
  import mujoco_warp as mjwarp

  entity = articulated_entity
  model = entity.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetData(model, data)

  wp_model = mjwarp.put_model(model)
  wp_data = mjwarp.put_data(model, data, nworld=1)

  wp_model = WarpBridge(wp_model)
  wp_data = WarpBridge(wp_data)

  entity.initialize(model, wp_model, wp_data, "cuda")
  return entity, wp_data


# ============================================================================
# Parameterized Tests for Entity Properties
# ============================================================================


class TestEntityProperties:
  """Test entity property detection based on configuration."""

  @pytest.mark.parametrize(
    "entity_fixture,expected_props",
    [
      (
        "fixed_base_entity",
        {"is_fixed_base": True, "is_articulated": False, "is_actuated": False},
      ),
      (
        "floating_base_entity",
        {"is_fixed_base": False, "is_articulated": False, "is_actuated": False},
      ),
      (
        "articulated_entity",
        {"is_fixed_base": False, "is_articulated": True, "is_actuated": True},
      ),
      (
        "fixed_articulated_entity",
        {"is_fixed_base": True, "is_articulated": True, "is_actuated": True},
      ),
    ],
  )
  def test_entity_type_properties(self, entity_fixture, expected_props, request):
    """Test that entity type properties are correctly identified."""
    entity = request.getfixturevalue(entity_fixture)

    for prop_name, expected_value in expected_props.items():
      actual_value = getattr(entity, prop_name)
      assert actual_value == expected_value, (
        f"Entity.{prop_name} should be {expected_value} for {entity_fixture}"
      )

  @pytest.mark.parametrize(
    "entity_fixture,expected_counts",
    [
      (
        "fixed_base_entity",
        {"num_bodies": 1, "num_geoms": 1, "num_joints": 0, "num_actuators": 0},
      ),
      (
        "floating_base_entity",
        {"num_bodies": 1, "num_geoms": 1, "num_joints": 0, "num_actuators": 0},
      ),
      (
        "articulated_entity",
        {"num_bodies": 3, "num_geoms": 3, "num_joints": 2, "num_actuators": 2},
      ),
    ],
  )
  def test_entity_element_counts(self, entity_fixture, expected_counts, request):
    """Test that element counts are correct."""
    entity = request.getfixturevalue(entity_fixture)

    for count_name, expected_value in expected_counts.items():
      actual_value = getattr(entity, count_name)
      assert actual_value == expected_value, (
        f"Entity.{count_name} should be {expected_value} for {entity_fixture}"
      )


# ============================================================================
# Tests for Find Methods
# ============================================================================


class TestFindMethods:
  """Test entity element finding methods."""

  @pytest.mark.parametrize(
    "method_name,search_key,expected_names",
    [
      ("find_bodies", "base", ["base"]),
      ("find_bodies", "link.*", ["link1", "link2"]),  # Regex test
      ("find_joints", "joint1", ["joint1"]),
      ("find_joints", "joint.*", ["joint1", "joint2"]),  # Regex test
      ("find_geoms", "link1_geom", ["link1_geom"]),
      ("find_sites", "site1", ["site1"]),
      ("find_sensors", "joint1_pos", ["joint1_pos"]),
    ],
  )
  def test_find_methods_basic(
    self, articulated_entity, method_name, search_key, expected_names
  ):
    """Test basic find functionality with exact and regex matches."""
    method = getattr(articulated_entity, method_name)
    indices, names = method(search_key)

    assert names == expected_names, (
      f"{method_name}('{search_key}') should return {expected_names}, got {names}"
    )
    assert len(indices) == len(expected_names), (
      "Number of indices should match number of names"
    )

  def test_find_with_subset(self, articulated_entity):
    """Test find methods with subset parameter."""
    # Should find joint1 in the subset
    indices, names = articulated_entity.find_joints(
      "joint1", joint_subset=["joint1", "joint2"]
    )
    assert names == ["joint1"], "Should find joint1 in subset"

    # Should raise error when searching for joint not in subset
    with pytest.raises(ValueError, match="Not all regular expressions are matched"):
      articulated_entity.find_joints("joint1", joint_subset=["joint2"])


# ============================================================================
# Tests for State Management
# ============================================================================


class TestStateManagement:
  """Test reading and writing entity states."""

  def test_floating_base_root_state_write(self, initialized_floating_entity):
    """Test successful root state write for floating base entity."""
    entity, data = initialized_floating_entity

    # Create test state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    root_state = torch.tensor(
      [
        [
          1.0,
          2.0,
          3.0,  # position
          1.0,
          0.0,
          0.0,
          0.0,  # quaternion (w,x,y,z)
          0.1,
          0.2,
          0.3,  # linear velocity
          0.0,
          0.0,
          0.1,
        ]  # angular velocity
      ],
      device="cuda",
    )

    # Write state
    entity.write_root_state_to_sim(root_state)

    # Verify position and orientation were set
    q_slice = entity.data.indexing.free_joint_q_adr
    actual_pose = data.qpos[:, q_slice]
    assert torch.allclose(actual_pose, root_state[:, :7], atol=1e-6), (
      "Root pose should match written values"
    )

    # Verify velocities were set
    v_slice = entity.data.indexing.free_joint_v_adr
    actual_vel = data.qvel[:, v_slice]
    assert torch.allclose(actual_vel, root_state[:, 7:], atol=1e-6), (
      "Root velocities should match written values"
    )

  def test_articulated_joint_state_write(self, initialized_articulated_entity):
    """Test successful joint state write for articulated entity."""
    entity, data = initialized_articulated_entity

    # Create test joint states
    joint_positions = torch.tensor([[0.5, -0.5]], device="cuda")  # 2 joints
    joint_velocities = torch.tensor([[0.1, -0.1]], device="cuda")

    # Write state
    entity.write_joint_state_to_sim(joint_positions, joint_velocities)

    # Verify positions were set
    q_slice = entity.data.indexing.joint_q_adr
    actual_pos = data.qpos[:, q_slice]
    assert torch.allclose(actual_pos, joint_positions, atol=1e-6), (
      "Joint positions should match written values"
    )

    # Verify velocities were set
    v_slice = entity.data.indexing.joint_v_adr
    actual_vel = data.qvel[:, v_slice]
    assert torch.allclose(actual_vel, joint_velocities, atol=1e-6), (
      "Joint velocities should match written values"
    )

  def test_joint_control_parameters(self, initialized_articulated_entity):
    """Test writing joint stiffness and damping."""
    entity, _ = initialized_articulated_entity

    # Test scalar stiffness/damping
    entity.write_joint_stiffness_to_sim(2.0)
    assert torch.all(entity.data.joint_stiffness == 2.0), (
      "All joint stiffness values should be set to 2.0"
    )

    entity.write_joint_damping_to_sim(0.5)
    assert torch.all(entity.data.joint_damping == 0.5), (
      "All joint damping values should be set to 0.5"
    )

    # Test tensor stiffness/damping for specific joints
    new_stiffness = torch.tensor([[3.0]], device="cuda")
    entity.write_joint_stiffness_to_sim(new_stiffness, joint_ids=slice(0, 1))
    assert entity.data.joint_stiffness[0, 0] == 3.0, (
      "First joint stiffness should be updated to 3.0"
    )

  @pytest.mark.parametrize(
    "write_method,error_match",
    [
      ("write_root_state_to_sim", "Cannot write root state for fixed-base entity"),
      ("write_root_link_pose_to_sim", "Cannot write root pose for fixed-base entity"),
      (
        "write_root_link_velocity_to_sim",
        "Cannot write root velocity for fixed-base entity",
      ),
    ],
  )
  def test_fixed_base_root_write_errors(
    self, fixed_base_entity, write_method, error_match
  ):
    """Test that fixed-base entities properly reject root state writes."""
    method = getattr(fixed_base_entity, write_method)
    dummy_tensor = torch.zeros(1, 13)  # Max size for root_state

    with pytest.raises(ValueError, match=error_match):
      method(dummy_tensor)


# ============================================================================
# Tests for External Forces
# ============================================================================


class TestExternalForces:
  """Test application of external forces and torques."""

  def test_external_force_application(self, initialized_floating_entity):
    """Test applying external forces and torques."""
    entity, data = initialized_floating_entity

    # Apply force and torque to the main body
    forces = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
    torques = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")

    entity.set_external_force_and_torque(forces, torques)

    # Check forces were applied (body_id=1 for the object body)
    body_id = entity.indexing.body_ids[0]
    assert torch.allclose(data.xfrc_applied[:, body_id, :3], forces), (
      "External force should be applied to body"
    )
    assert torch.allclose(data.xfrc_applied[:, body_id, 3:], torques), (
      "External torque should be applied to body"
    )

  def test_clear_state(self, initialized_floating_entity):
    """Test that clear_state properly resets forces."""
    entity, data = initialized_floating_entity

    # Apply some forces
    forces = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    entity.set_external_force_and_torque(forces, torch.zeros_like(forces))

    # Clear state
    entity.clear_state()

    # Verify forces are cleared
    assert torch.all(data.xfrc_applied == 0), "All external forces should be cleared"
    assert torch.all(data.qfrc_applied == 0), "All applied forces should be cleared"


# ============================================================================
# Integration Test
# ============================================================================


class TestIntegration:
  """Test complete entity workflows."""

  def test_full_entity_lifecycle(self, articulated_xml, actuator_cfg, tmp_path):
    """Test complete entity lifecycle from creation to simulation."""
    import mujoco_warp as mjwarp

    # Create entity with custom initial state
    init_state = EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 1.5),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.1, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.1),
      joint_pos={"joint1": 0.1, "joint2": -0.1},
      joint_vel={"joint1": 0.0, "joint2": 0.0},
    )

    entity_cfg = EntityCfg(
      spec_fn=lambda: mujoco.MjSpec.from_string(articulated_xml),
      articulation=actuator_cfg,
      init_state=init_state,
    )
    entity = Entity(entity_cfg)

    # Verify initial configuration
    assert entity.is_articulated, "Entity should be articulated"
    assert entity.is_actuated, "Entity should have actuators"

    # Write XML to disk
    xml_path = tmp_path / "test_entity.xml"
    entity.write_xml(xml_path)
    assert xml_path.exists(), "XML file should be written to disk"

    # Compile and initialize
    model = entity.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    wp_model = mjwarp.put_model(model)
    wp_data = mjwarp.put_data(model, data, nworld=2)  # Test with nworld=2.

    wp_model = WarpBridge(wp_model)
    wp_data = WarpBridge(wp_data)

    entity.initialize(model, wp_model, wp_data, "cuda")  # type: ignore

    # Test state writing
    root_state = torch.tensor(
      [
        [1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ],
      device="cuda",
    )
    entity.write_root_state_to_sim(root_state)

    joint_pos = torch.tensor([[0.2, -0.2], [0.3, -0.3]], device="cuda")
    joint_vel = torch.zeros_like(joint_pos)
    entity.write_joint_state_to_sim(joint_pos, joint_vel)

    # Apply forces to first environment only
    forces = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
    entity.set_external_force_and_torque(
      forces, torch.zeros_like(forces), env_ids=torch.tensor([0])
    )

    # Update simulation
    entity.update(0.01)

    # Reset second environment
    entity.reset(env_ids=torch.tensor([1]))

    # Verify second environment was reset
    assert torch.all(wp_data.xfrc_applied[1] == 0), (
      "Second environment should have forces cleared after reset"
    )

    # Test data properties
    assert entity.data.root_link_pose_w.shape == (2, 7), (
      "Root pose should have shape (num_envs, 7)"
    )
    assert entity.data.joint_pos.shape == (2, 2), (
      "Joint positions should have shape (num_envs, num_joints)"
    )


class TestEdgeCases:
  """Test edge cases and potential failure modes."""

  def test_entity_without_names(self):
    """Test that entities without names get auto-generated names."""
    xml = """
        <mujoco>
          <worldbody>
            <body pos="0 0 0.5">
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))
    entity = Entity(cfg)

    assert entity.body_names[0].startswith("body_"), "Should auto-generate body name"
    assert entity.geom_names[0].startswith("geom_"), "Should auto-generate geom name"


class TestEntityDataProperties:
  """Test EntityData property accessors."""

  def test_body_properties(self, initialized_articulated_entity):
    """Test body-related data properties."""
    entity, _ = initialized_articulated_entity
    data = entity.data

    # Test shapes
    assert data.body_link_pose_w.shape == (1, 3, 7)
    assert data.body_link_vel_w.shape == (1, 3, 6)
    assert data.body_com_pose_w.shape == (1, 3, 7)

    # Test decomposed properties
    assert data.body_link_pos_w.shape == (1, 3, 3)
    assert data.body_link_quat_w.shape == (1, 3, 4)

    # Test derived properties
    assert data.projected_gravity_b.shape == (1, 3)
    assert data.heading_w.shape == (1,)

  def test_sensor_data_access(self, initialized_articulated_entity):
    """Test that sensor data can be accessed."""
    entity, _ = initialized_articulated_entity

    # The fixture has a jointpos sensor
    assert "joint1_pos" in entity.data.indexing.sensor_adr
    sensor_adr = entity.data.indexing.sensor_adr["joint1_pos"]
    assert sensor_adr is not None


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
