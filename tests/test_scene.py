"""Tests for Scene class."""

from unittest.mock import Mock

import mujoco
import pytest
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim_data import WarpBridge

# ============================================================================
# Fixtures
# ============================================================================


def get_test_device():
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda:0"
  return "cpu"


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def simple_entity_xml():
  """Simple entity XML for testing."""
  return """
    <mujoco>
      <worldbody>
        <body name="box" pos="0 0 0.5">
          <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """


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


# ============================================================================
# Basic Scene Tests
# ============================================================================


class TestSceneInitialization:
  """Test Scene initialization and configuration."""

  def test_minimal_scene_creation(self, minimal_scene_cfg, device):
    """Test creating a minimal scene with no entities."""
    scene = Scene(minimal_scene_cfg, device)

    assert scene.num_envs == 1
    assert scene.env_spacing == 2.0
    assert len(scene.entities) == 0
    assert scene.terrain is None

  def test_scene_with_entities(self, scene_with_entities_cfg, device):
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


class TestSceneCompilation:
  """Test Scene compilation and XML generation."""

  def test_compile_empty_scene(self, minimal_scene_cfg, device):
    """Test compiling an empty scene."""
    scene = Scene(minimal_scene_cfg, device=device)
    model = scene.compile()

    assert isinstance(model, mujoco.MjModel)
    assert model.nbody == 1
    assert model.nq == model.nv == 0

  def test_compile_scene_with_entities(self, scene_with_entities_cfg, device):
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
  def test_to_zip(self, minimal_scene_cfg, tmp_path, device):
    """Test exporting scene to zip file."""
    scene = Scene(minimal_scene_cfg, device)
    zip_path = tmp_path / "scene.zip"

    scene.to_zip(zip_path)
    assert zip_path.exists()


# ============================================================================
# Entity Access Tests
# ============================================================================


class TestEntityAccess:
  """Test accessing entities in the scene."""

  def test_entity_dict_access(self, scene_with_entities_cfg, device):
    """Test accessing entities through dictionary."""
    scene = Scene(scene_with_entities_cfg, device)

    box = scene.entities["box"]
    robot = scene.entities["robot"]

    assert isinstance(box, Entity)
    assert isinstance(robot, Entity)
    assert box.is_fixed_base
    assert not robot.is_fixed_base

  def test_entity_getitem_access(self, scene_with_entities_cfg, device):
    """Test accessing entities through __getitem__."""
    scene = Scene(scene_with_entities_cfg, device)

    box = scene["box"]
    robot = scene["robot"]

    assert isinstance(box, Entity)
    assert isinstance(robot, Entity)

  def test_invalid_entity_access(self, scene_with_entities_cfg, device):
    """Test accessing non-existent entity raises KeyError."""
    scene = Scene(scene_with_entities_cfg, device)

    with pytest.raises(KeyError, match="Scene entity with key 'invalid' not found"):
      _ = scene["invalid"]


# ============================================================================
# Scene Initialization Tests
# ============================================================================


class TestSceneSimulationInitialization:
  """Test Scene initialization with simulation."""

  @pytest.fixture
  def initialized_scene(self, scene_with_entities_cfg, device):
    """Create an initialized scene with simulation."""
    import mujoco_warp as mjwarp

    scene = Scene(scene_with_entities_cfg, device)
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    wp_model = mjwarp.put_model(model)
    wp_data = mjwarp.put_data(model, data, nworld=scene.num_envs)
    wp_model = WarpBridge(wp_model, nworld=scene.num_envs)
    wp_data = WarpBridge(wp_data)

    scene.initialize(model, wp_model, wp_data, device)  # type: ignore
    return scene, wp_data

  def test_scene_initialize(self, initialized_scene, device):
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

  def test_env_origins_without_terrain(self, initialized_scene):
    """Test env_origins property without terrain."""
    scene, _ = initialized_scene

    origins = scene.env_origins
    assert origins.shape == (4, 3)
    assert torch.all(origins == 0)  # Default origins should be zeros.


# ============================================================================
# Scene Operations Tests
# ============================================================================


class TestSceneOperations:
  """Test Scene operations like reset, update, write_data_to_sim."""

  @pytest.fixture
  def mock_entities(self):
    """Create mock entities for testing."""
    mock_box = Mock(spec=Entity)
    mock_robot = Mock(spec=Entity)
    return {"box": mock_box, "robot": mock_robot}

  def test_scene_reset(self, minimal_scene_cfg, mock_entities, device):
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

  def test_scene_update(self, minimal_scene_cfg, mock_entities, device):
    """Test that update calls update on all entities."""
    scene = Scene(minimal_scene_cfg, device)
    scene._entities = mock_entities

    dt = 0.01
    scene.update(dt)

    for entity in mock_entities.values():
      entity.update.assert_called_once_with(dt)

  def test_scene_write_data_to_sim(self, minimal_scene_cfg, mock_entities, device):
    """Test that write_data_to_sim calls the method on all entities."""
    scene = Scene(minimal_scene_cfg, device)
    scene._entities = mock_entities

    scene.write_data_to_sim()

    for entity in mock_entities.values():
      entity.write_data_to_sim.assert_called_once()


# ============================================================================
# Integration Tests
# ============================================================================


class TestSceneIntegration:
  """Integration tests for complete scene workflows."""

  def test_full_scene_lifecycle(self, robot_entity_cfg, device, tmp_path):
    """Test complete scene lifecycle from creation to simulation."""
    import mujoco_warp as mjwarp

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

    scene.initialize(model, wp_model, wp_data, device)  # type: ignore

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


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
