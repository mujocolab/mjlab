"""Tests for camera_sensor.py."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor.camera_sensor import CameraSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture(scope="module")
def simple_world_xml():
  """XML for a simple world with a box."""
  return """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
        <body name="box" pos="0 0 0.5">
          <geom name="box_geom" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
        </body>
      </worldbody>
    </mujoco>
  """


@pytest.fixture(scope="module")
def world_with_camera_xml():
  """XML for world with existing camera."""
  return """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
        <body name="box" pos="0 0 0.5">
          <geom name="box_geom" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
          <camera name="box_cam" pos="0 0 1" quat="1 0 0 0"/>
        </body>
      </worldbody>
    </mujoco>
  """


def test_camera_sensor_rgb_output(simple_world_xml, device):
  """Verify camera sensor returns RGB data with correct shape."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera_cfg = CameraSensorCfg(
    name="test_cam",
    pos=(2.0, 0.0, 1.0),
    quat=(0.924, 0.0, 0.383, 0.0),
    width=64,
    height=48,
    type=("rgb",),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["test_cam"]
  sim.step()
  scene.update(sim.mj_model.opt.timestep)
  data = sensor.data

  assert data.rgb is not None
  assert data.depth is None
  assert data.rgb.shape == (2, 48, 64, 3)
  assert data.rgb.dtype == torch.uint8


def test_camera_sensor_depth_output(simple_world_xml, device):
  """Verify camera sensor returns depth data with correct shape."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera_cfg = CameraSensorCfg(
    name="test_cam",
    pos=(2.0, 0.0, 1.0),
    width=64,
    height=48,
    type=("depth",),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["test_cam"]
  sim.step()
  scene.update(sim.mj_model.opt.timestep)
  data = sensor.data

  assert data.depth is not None
  assert data.rgb is None
  assert data.depth.shape == (2, 48, 64, 1)
  assert data.depth.dtype == torch.float32


def test_camera_sensor_rgb_and_depth(simple_world_xml, device):
  """Verify camera sensor returns both RGB and depth when configured."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera_cfg = CameraSensorCfg(
    name="test_cam",
    pos=(2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("rgb", "depth"),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["test_cam"]
  sim.step()
  scene.update(sim.mj_model.opt.timestep)
  data = sensor.data

  assert data.rgb is not None
  assert data.depth is not None
  assert data.rgb.shape == (1, 24, 32, 3)
  assert data.depth.shape == (1, 24, 32, 1)


def test_camera_sensor_wraps_existing(world_with_camera_xml, device):
  """Verify camera sensor can wrap an existing camera from XML."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(world_with_camera_xml)
  )

  # Wrap the existing "box_cam" camera.
  camera_cfg = CameraSensorCfg(
    name="wrapped_cam",
    camera_name="world/box_cam",
    width=64,
    height=48,
    type=("rgb",),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["wrapped_cam"]
  sim.step()
  scene.update(sim.mj_model.opt.timestep)
  data = sensor.data

  assert data.rgb is not None
  assert data.rgb.shape == (1, 48, 64, 3)


def test_multiple_cameras_with_different_resolutions(simple_world_xml, device):
  """Verify multiple cameras with different resolutions work correctly."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera1_cfg = CameraSensorCfg(
    name="cam1",
    pos=(2.0, 0.0, 1.0),
    width=64,
    height=48,
    type=("rgb",),
  )

  camera2_cfg = CameraSensorCfg(
    name="cam2",
    pos=(-2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("rgb",),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera1_cfg, camera2_cfg),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor1 = scene["cam1"]
  sensor2 = scene["cam2"]

  sim.step()
  scene.update(sim.mj_model.opt.timestep)

  data1 = sensor1.data
  data2 = sensor2.data

  assert data1.rgb.shape == (2, 48, 64, 3)
  assert data2.rgb.shape == (2, 24, 32, 3)


def test_error_on_mismatched_render_settings(simple_world_xml, device):
  """Verify error when cameras have inconsistent render settings."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera1_cfg = CameraSensorCfg(
    name="cam1",
    pos=(2.0, 0.0, 1.0),
    width=64,
    height=48,
    type=("rgb",),
    use_textures=True,
  )

  camera2_cfg = CameraSensorCfg(
    name="cam2",
    pos=(-2.0, 0.0, 1.0),
    width=64,
    height=48,
    type=("rgb",),
    use_textures=False,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera1_cfg, camera2_cfg),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  with pytest.raises(ValueError, match="use_textures"):
    scene.initialize(sim.mj_model, sim.model, sim.data)


def test_error_on_invalid_camera_name(simple_world_xml, device):
  """Verify error when wrapping nonexistent camera."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera_cfg = CameraSensorCfg(
    name="bad_cam",
    camera_name="world/nonexistent_cam",
    width=64,
    height=48,
    type=("rgb",),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  with pytest.raises(ValueError, match="not found in model"):
    scene.initialize(sim.mj_model, sim.model, sim.data)


def test_camera_validation_mixed_types(simple_world_xml, device):
  """Verify per-camera validation when cameras have different types."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  # Camera with RGB only.
  rgb_cam_cfg = CameraSensorCfg(
    name="rgb_cam",
    pos=(2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("rgb",),
  )

  # Camera with depth only.
  depth_cam_cfg = CameraSensorCfg(
    name="depth_cam",
    pos=(-2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("depth",),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(rgb_cam_cfg, depth_cam_cfg),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sim.step()
  scene.update(sim.mj_model.opt.timestep)

  rgb_sensor = scene["rgb_cam"]
  depth_sensor = scene["depth_cam"]

  # RGB sensor should return RGB data, not depth.
  rgb_data = rgb_sensor.data
  assert rgb_data.rgb is not None
  assert rgb_data.depth is None

  # Depth sensor should return depth data, not RGB.
  depth_data = depth_sensor.data
  assert depth_data.depth is not None
  assert depth_data.rgb is None

  # Directly calling get_depth on RGB-only camera should fail.
  assert scene._render_manager is not None
  with pytest.raises(RuntimeError, match="does not have depth rendering enabled"):
    scene._render_manager.get_depth(rgb_sensor.camera_idx)

  # Directly calling get_rgb on depth-only camera should fail.
  with pytest.raises(RuntimeError, match="does not have RGB rendering enabled"):
    scene._render_manager.get_rgb(depth_sensor.camera_idx)


def test_lazy_rendering(simple_world_xml, device):
  """Verify lazy rendering only renders when data is accessed."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera_cfg = CameraSensorCfg(
    name="test_cam",
    pos=(2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("rgb",),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  render_manager = scene._render_manager
  assert render_manager is not None

  # After init, needs_render should be True.
  assert render_manager._needs_render

  # Step and update without accessing data.
  sim.step()
  scene.update(sim.mj_model.opt.timestep)

  # Should still need render (invalidated by update).
  assert render_manager._needs_render

  # Access data - this triggers rendering.
  sensor = scene["test_cam"]
  data = sensor.data
  assert data.rgb is not None

  # After data access, should not need render.
  assert not render_manager._needs_render

  # Another update invalidates again.
  sim.step()
  scene.update(sim.mj_model.opt.timestep)
  assert render_manager._needs_render


def test_clone_data_option(simple_world_xml, device):
  """Verify clone_data config option controls tensor cloning behavior."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  # Camera with clone_data=False (default).
  camera_no_clone_cfg = CameraSensorCfg(
    name="no_clone_cam",
    pos=(2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("rgb",),
    clone_data=False,
  )

  # Camera with clone_data=True.
  camera_clone_cfg = CameraSensorCfg(
    name="clone_cam",
    pos=(-2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("rgb",),
    clone_data=True,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_no_clone_cfg, camera_clone_cfg),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sim.step()
  scene.update(sim.mj_model.opt.timestep)

  no_clone_sensor = scene["no_clone_cam"]
  clone_sensor = scene["clone_cam"]

  # Get data multiple times.
  no_clone_data1 = no_clone_sensor.data
  no_clone_data2 = no_clone_sensor.data
  clone_data1 = clone_sensor.data
  clone_data2 = clone_sensor.data

  # With clone_data=False, data accesses return same cached object.
  assert no_clone_data1 is no_clone_data2

  # With clone_data=True, data accesses also return same cached object
  # (cloning happens at compute time, not on each access).
  assert clone_data1 is clone_data2

  # Both should have valid data.
  assert no_clone_data1.rgb is not None
  assert clone_data1.rgb is not None
