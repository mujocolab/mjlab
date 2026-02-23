"""Tests for camera_sensor.py and the unified sense pipeline."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import CameraSensorCfg, CameraSensorData
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


SCENE_WITH_CAMERA_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"
            rgba="0.5 0.5 0.5 1"/>
      <geom name="red_box" type="box" size="0.5 0.5 0.5" pos="0 0 0.5"
            rgba="1 0 0 1"/>
      <camera name="overhead_cam" pos="0 0 3" quat="1 0 0 0"
              fovy="45" resolution="32 24"/>
    </worldbody>
  </mujoco>
"""


def _make_scene_and_sim(
  device: str,
  sensors: tuple,
  xml: str = SCENE_WITH_CAMERA_XML,
  num_envs: int = 2,
) -> tuple[Scene, Simulation]:
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))
  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=sensors,
  )
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  if scene.sensor_context is not None:
    sim.set_sensor_context(scene.sensor_context)
  return scene, sim


def test_camera_rgb_shape(device):
  """RGB data has correct shape [num_envs, height, width, 3]."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    data_types=("rgb",),
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,))

  sim.forward()
  sim.sense()
  sensor = scene["test_cam"]
  data = sensor.data

  assert isinstance(data, CameraSensorData)
  assert data.rgb is not None
  assert data.rgb.shape == (2, 24, 32, 3)
  assert data.rgb.dtype == torch.uint8
  assert data.depth is None


def test_camera_depth_shape(device):
  """Depth data has correct shape [num_envs, height, width]."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    data_types=("depth",),
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,))

  sim.forward()
  sim.sense()
  data = scene["test_cam"].data

  assert isinstance(data, CameraSensorData)
  assert data.depth is not None
  assert data.depth.shape == (2, 24, 32, 1)
  assert data.depth.dtype == torch.float32
  assert data.rgb is None


def test_camera_rgb_and_depth(device):
  """Both RGB and depth can be requested simultaneously."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    data_types=("rgb", "depth"),
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,))

  sim.forward()
  sim.sense()
  data = scene["test_cam"].data

  assert data.rgb is not None
  assert data.depth is not None
  assert data.rgb.shape == (2, 24, 32, 3)
  assert data.depth.shape == (2, 24, 32, 1)


def test_camera_create_new(device):
  """Camera sensor can create a new camera (not wrap existing)."""
  # XML without any cameras.
  xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="10 10 0.1"
              rgba="0.5 0.5 0.5 1"/>
      </worldbody>
    </mujoco>
  """
  cam_cfg = CameraSensorCfg(
    name="my_camera",
    pos=(0.0, 0.0, 3.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    fovy=45.0,
    width=16,
    height=12,
    data_types=("rgb",),
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,), xml=xml)

  sim.forward()
  sim.sense()
  data = scene["my_camera"].data

  assert data.rgb is not None
  assert data.rgb.shape == (2, 12, 16, 3)


def test_camera_rgb_not_all_zeros(device):
  """RGB data should contain non-zero values when scene has objects."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    data_types=("rgb",),
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,))

  sim.forward()
  sim.sense()
  data = scene["test_cam"].data

  assert data.rgb is not None
  # At least some pixels should be non-zero (scene has colored objects).
  assert data.rgb.any(), "RGB data is all zeros - rendering may have failed"


def test_camera_wrap_existing_overrides_fovy(device):
  """Wrapping an existing camera can override its fovy."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    fovy=90.0,  # Override the camera's fovy
    data_types=("rgb",),
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,))

  # Verify the camera's fovy was updated.
  cam_id = sim.mj_model.camera("world/overhead_cam").id
  assert sim.mj_model.cam_fovy[cam_id] == pytest.approx(90.0)

  sim.forward()
  sim.sense()
  data = scene["test_cam"].data
  assert data.rgb is not None


@pytest.mark.skipif(
  not torch.cuda.is_available(), reason="CUDA required for sense graph"
)
def test_sense_graph_captured(device):
  """Verify sense_graph is captured on CUDA."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=16,
    height=12,
    data_types=("rgb",),
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,))

  # On CUDA with mempool, sense_graph should be captured.
  if sim.use_cuda_graph:
    assert sim.sense_graph is not None


def test_camera_clone_data(device):
  """clone_data=True returns independent tensors."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=16,
    height=12,
    data_types=("rgb",),
    clone_data=True,
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,))

  sim.forward()
  sim.sense()
  data1 = scene["test_cam"].data
  rgb1 = data1.rgb

  # Invalidate cache and re-read.
  scene["test_cam"]._invalidate_cache()
  data2 = scene["test_cam"].data
  rgb2 = data2.rgb

  # With clone_data=True, the tensors should be different objects.
  assert rgb1 is not None and rgb2 is not None
  assert rgb1.data_ptr() != rgb2.data_ptr()


def test_camera_sensor_context_created(device):
  """Scene creates SensorContext when camera sensors are present."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=16,
    height=12,
    data_types=("rgb",),
  )
  scene, _ = _make_scene_and_sim(device, sensors=(cam_cfg,))

  assert scene.sensor_context is not None
  assert scene.sensor_context.has_cameras


INTRINSIC_CAM_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 5" dir="0 0 -1"/>
      <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"
            rgba="0.5 0.5 0.5 1"/>
      <geom name="small_box" type="box" size="0.1 0.1 0.1" pos="0 0 0.1"
            rgba="1 0 0 1"/>
      <camera name="intrinsic_cam" pos="0 0 5" quat="1 0 0 0"
              resolution="32 24" sensorsize="0.01 0.01"
              focal="0.005 0.005"/>
    </worldbody>
  </mujoco>
"""


def test_expand_cam_intrinsic_disables_precomputed_rays(device):
  """Expanding cam_intrinsic disables precomputed rays on the render context."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/intrinsic_cam",
    width=32,
    height=24,
    data_types=("rgb",),
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,), xml=INTRINSIC_CAM_XML)
  assert scene.sensor_context is not None

  # Before expansion: precomputed rays enabled.
  rc = scene.sensor_context.render_context
  assert rc.use_precomputed_rays is True

  sim.expand_model_fields(("cam_intrinsic",))

  # After expansion: precomputed rays disabled.
  rc = scene.sensor_context.render_context
  assert rc.use_precomputed_rays is False


def test_cam_intrinsic_dr_changes_rendered_image(device):
  """Per-env cam_intrinsic DR produces different images across envs."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/intrinsic_cam",
    width=32,
    height=24,
    data_types=("rgb",),
  )
  scene, sim = _make_scene_and_sim(
    device, sensors=(cam_cfg,), xml=INTRINSIC_CAM_XML, num_envs=2
  )
  sim.expand_model_fields(("cam_intrinsic",))

  # Render baseline.
  sim.forward()
  sim.sense()
  baseline = scene["test_cam"].data.rgb.clone()

  # Set env 0 to a very different focal length (5x zoom).
  cam_id = sim.mj_model.camera("world/intrinsic_cam").id
  sim.model.cam_intrinsic[0, cam_id, :2] *= 5.0

  sim.forward()
  sim.sense()
  after = scene["test_cam"].data.rgb

  # Env 0 should differ (zoomed in), env 1 should be unchanged.
  assert not torch.equal(after[0], baseline[0]), (
    "Env 0 image unchanged after cam_intrinsic modification"
  )
  assert torch.equal(after[1], baseline[1]), "Env 1 image changed unexpectedly"


def test_camera_create_on_parent_body(device):
  """Camera sensor can create a new camera attached to a body."""
  xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="10 10 0.1"
              rgba="0.5 0.5 0.5 1"/>
        <body name="arm_link" pos="0 0 1">
          <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
        </body>
      </worldbody>
    </mujoco>
  """
  cam_cfg = CameraSensorCfg(
    name="wrist_cam",
    parent_body="world/arm_link",
    pos=(0.0, 0.0, 0.2),
    quat=(1.0, 0.0, 0.0, 0.0),
    fovy=45.0,
    width=16,
    height=12,
    data_types=("rgb",),
  )
  scene, sim = _make_scene_and_sim(device, sensors=(cam_cfg,), xml=xml)

  sim.forward()
  sim.sense()
  data = scene["wrist_cam"].data

  assert isinstance(data, CameraSensorData)
  assert data.rgb is not None
  assert data.rgb.shape == (2, 12, 16, 3)
