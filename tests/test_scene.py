"""Tests for Scene class."""

from unittest.mock import Mock

import mujoco
import mujoco_warp as mjwarp
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.sim.sim_data import WarpBridge

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def simple_entity_xml():
  """Simple entity XML for testing."""
  return load_fixture_xml("fixed_base_box")


@pytest.fixture
def robot_entity_xml():
  """Robot entity XML for testing."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
          <body name="link1" pos="0.3 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def simple_entity_cfg(simple_entity_xml):
  """Entity config for a simple box."""
  return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_entity_xml))


@pytest.fixture
def robot_entity_cfg(robot_entity_xml):
  """Entity config for a robot."""
  return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(robot_entity_xml))


@pytest.fixture
def minimal_scene_cfg():
  """Minimal scene configuration."""
  return SceneCfg(
    num_envs=1,
    env_spacing=2.0,
  )


@pytest.fixture
def scene_with_entities_cfg(simple_entity_cfg, robot_entity_cfg):
  """Scene configuration with multiple entities."""
  return SceneCfg(
    num_envs=4,
    env_spacing=3.0,
    entities={
      "box": simple_entity_cfg,
      "robot": robot_entity_cfg,
    },
  )


@pytest.fixture
def entity_with_site_xml():
  """Entity XML with a site for tendon attachment."""
  return """
    <mujoco>
      <worldbody>
        <body name="box" pos="0 0 0.5">
          <freejoint name="free"/>
          <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
          <site name="hook" pos="0 0 0.1" size="0.01"/>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def initialized_scene(scene_with_entities_cfg, device):
  """Create an initialized scene with simulation."""
  scene = Scene(scene_with_entities_cfg, device)
  model = scene.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetData(model, data)

  wp_model = mjwarp.put_model(model)
  wp_data = mjwarp.put_data(model, data, nworld=scene.num_envs)
  wp_model = WarpBridge(wp_model, nworld=scene.num_envs)
  wp_data = WarpBridge(wp_data)

  scene.initialize(model, wp_model, wp_data)  # type: ignore
  return scene, wp_data


@pytest.fixture
def mock_entities():
  """Create mock entities for testing."""
  mock_box = Mock(spec=Entity)
  mock_robot = Mock(spec=Entity)
  return {"box": mock_box, "robot": mock_robot}


# ============================================================================
# Basic Scene Tests
# ============================================================================


def test_minimal_scene_creation(minimal_scene_cfg, device):
  """Test creating a minimal scene with no entities."""
  scene = Scene(minimal_scene_cfg, device)

  assert scene.num_envs == 1
  assert scene.env_spacing == 2.0
  assert len(scene.entities) == 0
  assert scene.terrain is None


def test_scene_with_entities(scene_with_entities_cfg, device):
  """Test creating a scene with multiple entities."""
  scene = Scene(scene_with_entities_cfg, device)

  assert scene.num_envs == 4
  assert scene.env_spacing == 3.0
  assert len(scene.entities) == 2
  assert "box" in scene.entities
  assert "robot" in scene.entities
  assert isinstance(scene.entities["box"], Entity)
  assert isinstance(scene.entities["robot"], Entity)


# ============================================================================
# Scene Compilation Tests
# ============================================================================


def test_compile_empty_scene(minimal_scene_cfg, device):
  """Test compiling an empty scene."""
  scene = Scene(minimal_scene_cfg, device=device)
  model = scene.compile()

  assert isinstance(model, mujoco.MjModel)
  assert model.nbody == 1
  assert model.nq == model.nv == 0


def test_compile_scene_with_entities(scene_with_entities_cfg, device):
  """Test compiling a scene with entities."""
  scene = Scene(scene_with_entities_cfg, device)
  model = scene.compile()

  assert isinstance(model, mujoco.MjModel)
  # Should have world + entity bodies.
  assert model.nbody > 1
  # Check that entity names are prefixed
  body_names = [model.body(i).name for i in range(model.nbody)]
  assert any("box/" in name for name in body_names)
  assert any("robot/" in name for name in body_names)


# TODO: Test that we can unzip and reload the scene correctly.
def test_to_zip(minimal_scene_cfg, tmp_path, device):
  """Test exporting scene to zip file."""
  scene = Scene(minimal_scene_cfg, device)
  zip_path = tmp_path / "scene.zip"

  scene.to_zip(zip_path)
  assert zip_path.exists()


# ============================================================================
# Entity Access Tests
# ============================================================================


def test_entity_dict_access(scene_with_entities_cfg, device):
  """Test accessing entities through dictionary."""
  scene = Scene(scene_with_entities_cfg, device)

  box = scene.entities["box"]
  robot = scene.entities["robot"]

  assert isinstance(box, Entity)
  assert isinstance(robot, Entity)
  assert box.is_fixed_base
  assert not robot.is_fixed_base


def test_entity_getitem_access(scene_with_entities_cfg, device):
  """Test accessing entities through __getitem__."""
  scene = Scene(scene_with_entities_cfg, device)

  box = scene["box"]
  robot = scene["robot"]

  assert isinstance(box, Entity)
  assert isinstance(robot, Entity)


def test_invalid_entity_access(scene_with_entities_cfg, device):
  """Test accessing non-existent entity raises KeyError."""
  scene = Scene(scene_with_entities_cfg, device)

  with pytest.raises(KeyError, match="Scene element 'invalid' not found"):
    _ = scene["invalid"]


# ============================================================================
# Scene Initialization Tests
# ============================================================================


def test_scene_initialize(initialized_scene, device):
  """Test that scene initialization sets up entities."""
  scene, _ = initialized_scene

  # Check default env origins are set.
  assert scene._default_env_origins is not None
  assert scene._default_env_origins.shape == (4, 3)  # 4 envs, 3D positions.
  assert scene._default_env_origins.device.type == device.split(":")[0]

  # Check entities are initialized.
  for entity in scene.entities.values():
    assert hasattr(entity, "data")
    assert entity.data is not None


def test_env_origins_without_terrain(initialized_scene):
  """Test env_origins property without terrain."""
  scene, _ = initialized_scene

  origins = scene.env_origins
  assert origins.shape == (4, 3)
  assert torch.all(origins == 0)  # Default origins should be zeros.


# ============================================================================
# Scene Operations Tests
# ============================================================================


def test_scene_reset(minimal_scene_cfg, mock_entities, device):
  """Test that reset calls reset on all entities."""
  scene = Scene(minimal_scene_cfg, device)
  scene._entities = mock_entities

  # Reset all environments
  scene.reset()
  for entity in mock_entities.values():
    entity.reset.assert_called_once_with(None)

  # Reset specific environments
  for entity in mock_entities.values():
    entity.reset.reset_mock()

  env_ids = torch.tensor([0, 2])
  scene.reset(env_ids)
  for entity in mock_entities.values():
    entity.reset.assert_called_once_with(env_ids)


def test_scene_update(minimal_scene_cfg, mock_entities, device):
  """Test that update calls update on all entities."""
  scene = Scene(minimal_scene_cfg, device)
  scene._entities = mock_entities

  dt = 0.01
  scene.update(dt)

  for entity in mock_entities.values():
    entity.update.assert_called_once_with(dt)


def test_scene_write_data_to_sim(minimal_scene_cfg, mock_entities, device):
  """Test that write_data_to_sim calls the method on all entities."""
  scene = Scene(minimal_scene_cfg, device)
  scene._entities = mock_entities

  scene.write_data_to_sim()

  for entity in mock_entities.values():
    entity.write_data_to_sim.assert_called_once()


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_scene_lifecycle(robot_entity_cfg, device, tmp_path):
  """Test complete scene lifecycle from creation to simulation."""
  scene_cfg = SceneCfg(
    num_envs=3,
    env_spacing=2.5,
    entities={
      "robot1": robot_entity_cfg,
      "robot2": robot_entity_cfg,
    },
  )

  scene = Scene(scene_cfg, device)

  assert scene.num_envs == 3
  assert len(scene.entities) == 2

  model = scene.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetData(model, data)

  wp_model = mjwarp.put_model(model)
  wp_data = mjwarp.put_data(model, data, nworld=scene.num_envs)
  wp_model = WarpBridge(wp_model, nworld=scene.num_envs)
  wp_data = WarpBridge(wp_data)

  scene.initialize(model, wp_model, wp_data)  # type: ignore

  scene.reset()
  scene.update(0.01)
  scene.write_data_to_sim()

  scene.reset(env_ids=torch.tensor([0, 2]))

  zip_path = tmp_path / "test_scene.zip"
  scene.to_zip(zip_path)
  assert zip_path.exists()

  for entity in scene.entities.values():
    assert entity.data is not None
    if not entity.is_fixed_base:
      assert entity.data.root_link_pose_w.shape == (3, 7)


# ============================================================================
# Scene spec_fn Tests
# ============================================================================


def test_scene_spec_fn_adds_site(device):
  """Test that spec_fn can add elements to the scene spec."""

  def add_custom_site(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_site(name="custom_site", pos=(1, 2, 3))

  cfg = SceneCfg(spec_fn=add_custom_site)
  scene = Scene(cfg, device)
  model = scene.compile()

  site_id = model.site("custom_site").id
  assert site_id >= 0
  assert tuple(model.site_pos[site_id]) == (1.0, 2.0, 3.0)


def test_scene_spec_fn_cross_entity_tendon(entity_with_site_xml, device):
  """Test that spec_fn can create tendons between two entities."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(entity_with_site_xml)
  )

  def add_tendon_between_entities(spec: mujoco.MjSpec) -> None:
    # Add a world anchor site.
    spec.worldbody.add_site(name="anchor", pos=(0, 0, 2), size=(0.01,) * 3)
    # Create tendon from anchor to entity A.
    tendon_a = spec.add_tendon(name="rope_a", width=0.005)
    tendon_a.wrap_site("anchor")
    tendon_a.wrap_site("entity_a/hook")
    # Create tendon from anchor to entity B.
    tendon_b = spec.add_tendon(name="rope_b", width=0.005)
    tendon_b.wrap_site("anchor")
    tendon_b.wrap_site("entity_b/hook")

  cfg = SceneCfg(
    entities={"entity_a": entity_cfg, "entity_b": entity_cfg},
    spec_fn=add_tendon_between_entities,
  )
  scene = Scene(cfg, device)
  model = scene.compile()

  # Verify tendons exist.
  assert model.tendon("rope_a").id >= 0
  assert model.tendon("rope_b").id >= 0
  # Verify sites are referenced correctly (2 wraps per tendon).
  assert model.ntendon == 2
  assert model.nwrap == 4


# ============================================================================
# Keyframe Merging Tests
# ============================================================================


@pytest.fixture
def floating_box_cfg():
  """Entity config for a floating box with initial position."""
  xml = """
    <mujoco>
      <worldbody>
        <body name="box">
          <freejoint name="box_joint"/>
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
    </mujoco>
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(1.0, 2.0, 3.0)),
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
  )


@pytest.fixture
def floating_sphere_cfg():
  """Entity config for a floating sphere with initial position."""
  xml = """
    <mujoco>
      <worldbody>
        <body name="sphere">
          <freejoint name="sphere_joint"/>
          <geom type="sphere" size="0.1"/>
        </body>
      </worldbody>
    </mujoco>
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(4.0, 5.0, 6.0)),
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
  )


def test_single_entity_keyframe(floating_box_cfg, device):
  """Test that a single entity produces one merged keyframe."""
  cfg = SceneCfg(entities={"box": floating_box_cfg})
  scene = Scene(cfg, device)
  model = scene.compile()

  assert model.nkey == 1
  assert model.key(0).name == "init_state"
  # qpos: [x, y, z, qw, qx, qy, qz]
  assert tuple(model.key(0).qpos[:3]) == (1.0, 2.0, 3.0)


def test_multiple_entities_merged_keyframe(
  floating_box_cfg, floating_sphere_cfg, device
):
  """Test that multiple entities produce a single merged keyframe."""
  cfg = SceneCfg(entities={"box": floating_box_cfg, "sphere": floating_sphere_cfg})
  scene = Scene(cfg, device)
  model = scene.compile()

  assert model.nkey == 1
  assert model.key(0).name == "init_state"
  # Box qpos (0-6), sphere qpos (7-13).
  qpos = model.key(0).qpos
  assert tuple(qpos[:3]) == (1.0, 2.0, 3.0)  # box position
  assert tuple(qpos[7:10]) == (4.0, 5.0, 6.0)  # sphere position


# ============================================================================
# Multi-Entity Actuator Tests
# ============================================================================


def test_two_actuated_entities_write_ctrl(device):
  """Test that two identical actuated entities write controls to the correct global positions."""

  robot_xml = load_fixture_xml("floating_base_articulated")

  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("joint.*",),
          effort_limit=100.0,
          stiffness=80.0,
          damping=10.0,
        ),
      )
    ),
  )

  num_envs = 2
  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=3.0,
    entities={"robot_a": entity_cfg, "robot_b": entity_cfg},
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)  # type: ignore

  robot_a = scene["robot_a"]
  robot_b = scene["robot_b"]
  assert isinstance(robot_a, Entity)
  assert isinstance(robot_b, Entity)

  # Set different position targets for each entity.
  target_a = torch.tensor([[0.1, 0.2]], device=device).expand(num_envs, -1)
  target_b = torch.tensor([[0.5, 0.6]], device=device).expand(num_envs, -1)

  robot_a.set_joint_position_target(target_a)
  robot_a.set_joint_velocity_target(torch.zeros(num_envs, 2, device=device))
  robot_a.set_joint_effort_target(torch.zeros(num_envs, 2, device=device))

  robot_b.set_joint_position_target(target_b)
  robot_b.set_joint_velocity_target(torch.zeros(num_envs, 2, device=device))
  robot_b.set_joint_effort_target(torch.zeros(num_envs, 2, device=device))

  scene.write_data_to_sim()

  # Verify that each entity's controls landed in the correct global ctrl positions.
  global_ctrl_a = robot_a.indexing.ctrl_ids
  global_ctrl_b = robot_b.indexing.ctrl_ids

  # The two entities should have different global ctrl ranges.
  assert not torch.equal(global_ctrl_a, global_ctrl_b)

  # Check that the ctrl values in global positions match each entity's expected output.
  ctrl_a = sim.data.ctrl[0, global_ctrl_a]
  ctrl_b = sim.data.ctrl[0, global_ctrl_b]

  # Controls should differ since targets differ.
  assert not torch.allclose(ctrl_a, ctrl_b)
