"""Selected-world scene transfers and camera frame delivery regressions."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
import torch
import viser.transforms as vtf

from mjlab.sensor import CameraSensorCfg
from mjlab.sensor.sensor_context import SensorContext
from mjlab.sim.sim import Simulation
from mjlab.viewer.viser.camera_viewer import ViserCameraViewer
from mjlab.viewer.viser.overlays import ViserCameraOverlays
from mjlab.viewer.viser.scene import MjlabViserScene, _PerWorldMeshGroup
from mjlab.viewer.viser.viewer import ViserPlayViewer


class _TrackedTensor:
  """Record the tensor shape at the CPU transfer, after any indexing."""

  def __init__(self, tensor: torch.Tensor, copies: list[tuple[int, ...]]):
    self.tensor = tensor
    self.copies = copies

  def __getitem__(self, index: Any) -> _TrackedTensor:
    return _TrackedTensor(self.tensor[index], self.copies)

  def cpu(self) -> torch.Tensor:
    self.copies.append(tuple(self.tensor.shape))
    return self.tensor.cpu()


def _server() -> Any:
  server = MagicMock()

  def handle(*_args, **kwargs):
    return SimpleNamespace(visible=kwargs.get("visible", True), remove=MagicMock())

  for method in (
    "add_frame",
    "add_grid",
    "add_mesh_trimesh",
    "add_batched_meshes_trimesh",
    "add_batched_meshes_simple",
  ):
    getattr(server.scene, method).side_effect = handle
  return server


@pytest.fixture
def scene_frames():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="robot"><freejoint/><geom type="box" size=".1 .2 .3"/>
        <site name="marker" pos=".1 0 0"/>
        <body pos="0 0 .4"><joint name="hinge"/>
          <geom type="capsule" size=".05 .1"/>
        </body>
      </body>
      <body name="target" mocap="true"><geom size=".05"/></body>
    </worldbody><actuator><motor joint="hinge"/></actuator></mujoco>
  """)
  arrays: dict[str, list[np.ndarray]] = {
    name: []
    for name in ("xpos", "xmat", "mocap_pos", "mocap_quat", "qpos", "qvel", "ctrl")
  }
  for world in range(4):
    data = mujoco.MjData(model)
    data.qpos[0] = 10 * world
    data.qpos[-1] = 0.2 * world
    data.qvel[:] = 0.1 * world
    data.ctrl[:] = 0.3 * world
    data.mocap_pos[:] = [world, 2, 3]
    mujoco.mj_forward(model, data)
    for name, values in arrays.items():
      value = getattr(data, name).copy()
      values.append(value.reshape(model.nbody, 3, 3) if name == "xmat" else value)
  tensors = {
    name: torch.from_numpy(np.stack(values)).float() for name, values in arrays.items()
  }
  return model, tensors


@pytest.mark.parametrize("per_world", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_scene_selects_before_cpu_and_rotation_conversion(
  scene_frames, per_world: bool, selected: bool, monkeypatch: pytest.MonkeyPatch
):
  model, tensors = scene_frames
  sim_model = SimpleNamespace(
    geom_rgba=torch.from_numpy(model.geom_rgba.copy()).repeat(4, 1, 1),
    geom_dataid=torch.from_numpy(model.geom_dataid.copy()).repeat(4, 1),
  )
  sim_model.geom_rgba[:, :, 0] = torch.arange(4)[:, None] / 4
  scene = MjlabViserScene(
    _server(),
    model,
    4,
    sim_model=sim_model if per_world else None,
    expanded_fields={"geom_rgba"} if per_world else None,
  )
  scene.env_idx = 3
  scene.show_only_selected = selected
  scene.camera_tracking_enabled = True
  scene._tracked_body_id = 1
  monkeypatch.setattr(scene, "_any_decor_visible", lambda: True)
  monkeypatch.setattr(scene, "_update_decor_from_mjvscene", MagicMock())
  copies: list[tuple[int, ...]] = []
  data = SimpleNamespace(
    **{name: _TrackedTensor(tensor, copies) for name, tensor in tensors.items()}
  )
  rotations: list[tuple[int, ...]] = []
  from_matrix = vtf.SO3.from_matrix

  def record_rotations(matrices: np.ndarray):
    rotations.append(matrices.shape)
    return from_matrix(matrices)

  monkeypatch.setattr(vtf.SO3, "from_matrix", record_rotations)
  scene.update(data)
  expected_worlds = 1 if selected else 4
  assert len(copies) == len(tensors)
  assert all(shape[0] == expected_worlds for shape in copies)
  assert rotations == [(expected_worlds, model.nbody, 3, 3)]
  assert scene.env_idx == 3
  np.testing.assert_allclose(scene.mj_data.qpos, tensors["qpos"][3])
  np.testing.assert_allclose(scene.mj_data.qvel, tensors["qvel"][3])
  np.testing.assert_allclose(scene.mj_data.ctrl, tensors["ctrl"][3])
  np.testing.assert_allclose(scene._scene_offset, -tensors["xpos"][3, 1])
  if per_world:
    groups = [g for g in scene._mesh_groups if isinstance(g, _PerWorldMeshGroup)]
    assert groups
    for group in groups:
      mask = group.env_ids == 3 if selected else np.ones(len(group.env_ids), dtype=bool)
      assert group.handle.visible == bool(mask.any())
      if mask.any():
        expected = tensors["xpos"][group.env_ids[mask], group.body_ids[mask]].numpy()
        if group.mocap_ids is not None:
          expected = tensors["mocap_pos"][
            group.env_ids[mask], group.mocap_ids[mask]
          ].numpy()
        np.testing.assert_allclose(
          group.handle.batched_positions, expected + scene._scene_offset
        )


def test_compact_scene_selection_and_cached_refresh(scene_frames):
  model, tensors = scene_frames
  scene = MjlabViserScene(_server(), model, 4)
  data = SimpleNamespace(**tensors)
  scene.show_only_selected = True
  for world in (3, 1):
    scene.env_idx = world
    scene.update(data)
    assert scene._last_body_xpos is not None
    assert scene._last_body_xpos.shape[0] == 1
    np.testing.assert_array_equal(scene._last_body_xpos[0], tensors["xpos"][world])
    scene.refresh_visualization()
  scene.env_idx = 2
  scene.refresh_visualization()
  assert scene.needs_update
  scene.show_only_selected = False
  scene.refresh_visualization()  # Must not index absent worlds in the compact cache.
  assert scene.needs_update
  scene.update(data)
  assert scene._last_body_xpos is not None
  assert scene._last_body_xpos.shape[0] == 4
  scene.refresh_visualization()


def test_compact_scene_explicit_environment_keeps_ui_selection(scene_frames):
  model, tensors = scene_frames
  scene = MjlabViserScene(_server(), model, 4)
  scene.show_only_selected = True
  scene.update(SimpleNamespace(**tensors), env_idx=2)
  assert scene.env_idx == 0
  scene.refresh_visualization()
  assert not scene.needs_update
  assert scene._last_body_xpos is not None
  np.testing.assert_array_equal(scene._last_body_xpos[0], tensors["xpos"][2])
  scene.env_idx = 1
  scene.refresh_visualization()
  assert scene.needs_update


class _Image:
  def __init__(self, image: np.ndarray):
    self._image = image
    self.writes = 0

  @property
  def image(self) -> np.ndarray:
    return self._image

  @image.setter
  def image(self, image: np.ndarray) -> None:
    self._image = image
    self.writes += 1


class _Control:
  def __init__(self, value: Any):
    self.value = value
    self.callbacks: list[Callable[..., None]] = []

  def on_update(self, callback: Callable[..., None]) -> None:
    self.callbacks.append(callback)

  def set_value(self, value: Any) -> None:
    self.value = value
    for callback in self.callbacks:
      callback(None)


@pytest.fixture
def camera_viewer():
  server = _server()
  images: list[_Image] = []
  controls: dict[str, _Control] = {}

  def image(*, image, **_kwargs):
    handle = _Image(image)
    images.append(handle)
    return handle

  def control(*, label, initial_value, **_kwargs):
    handle = _Control(initial_value)
    controls[label] = handle
    return handle

  server.gui.add_image.side_effect = image
  server.gui.add_slider.side_effect = control
  server.gui.add_checkbox.side_effect = control
  copies: list[tuple[int, ...]] = []
  rgb = torch.zeros((2, 1, 2, 3), dtype=torch.uint8)
  rgb[1] = 255
  depth = torch.full((2, 1, 2, 1), 0.5)
  seg = torch.zeros((2, 1, 2, 2), dtype=torch.int32)
  sensor = CameraSensorCfg(
    name="cam",
    camera_name="cam",
    width=2,
    height=1,
    data_types=("rgb", "depth", "segmentation"),
  ).build()
  sensor._camera_idx = 0
  sensor.set_context(
    cast(
      Any,
      SimpleNamespace(
        get_rgb=lambda _: _TrackedTensor(rgb, copies),
        get_depth=lambda _: _TrackedTensor(depth, copies),
        get_segmentation=lambda _: _TrackedTensor(seg, copies),
      ),
    )
  )
  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><camera name="cam"/></worldbody></mujoco>'
  )
  viewer = ViserCameraViewer(server, sensor, model, min_display_size=1)
  sim_data = SimpleNamespace(
    cam_xpos=torch.zeros((2, 1, 3)),
    cam_xmat=torch.eye(3).repeat(2, 1, 1, 1),
  )
  return viewer, sensor, images, controls, copies, sim_data


def test_camera_reuses_unchanged_frames_and_refreshes_environment(camera_viewer):
  viewer, sensor, images, _controls, copies, data = camera_viewer
  viewer.update(data)
  assert [image.writes for image in images] == [1, 1, 1]
  assert len(copies) == 3
  assert not viewer.needs_update
  for _ in range(5):
    viewer.update(data)
  assert len(copies) == 3
  assert [image.writes for image in images] == [1, 1, 1]
  viewer.update(data, env_idx=1)
  assert len(copies) == 6
  assert [image.writes for image in images] == [2, 2, 2]
  assert images[0].image.min() == 255
  sensor._on_frame_ready()
  assert viewer.needs_update
  viewer.update(data, env_idx=1)
  assert len(copies) == 9
  assert [image.writes for image in images] == [3, 3, 3]


def test_camera_controls_refresh_while_paused_without_frame_copies(camera_viewer):
  camera, _sensor, images, controls, copies, data = camera_viewer
  camera.update(data)
  overlays = ViserCameraOverlays(MagicMock(), MagicMock(), MagicMock(), [camera])
  viewer = ViserPlayViewer(MagicMock(), MagicMock(), viser_server=MagicMock())
  viewer._camera_overlays = overlays
  viewer._scene = cast(
    MjlabViserScene, SimpleNamespace(env_idx=0, _scene_offset=np.zeros(3))
  )
  viewer._is_paused = True
  sim = MagicMock(spec=Simulation)
  sim.data = data
  controls["Depth Scale"].set_value(2.0)
  assert overlays.needs_update_for(0, np.zeros(3))
  viewer._update_camera_feeds(sim, has_pending_updates=False)
  assert not overlays.needs_update_for(0, np.zeros(3))
  assert [image.writes for image in images] == [1, 2, 1]
  assert len(copies) == 3
  assert images[1].image.shape == (1, 2, 3)
  assert images[1].image.max() == 63
  frustum = camera._frustum_handle
  assert frustum is not None
  controls["Frustum"].set_value(False)
  viewer._update_camera_feeds(sim, has_pending_updates=False)
  assert camera._frustum_handle is None
  cast(MagicMock, frustum.remove).assert_called_once()
  controls["Frustum"].set_value(True)
  viewer._update_camera_feeds(sim, has_pending_updates=False)
  assert camera._frustum_handle is not None
  assert not overlays.needs_update_for(0, np.zeros(3))
  assert [image.writes for image in images] == [1, 2, 1]
  assert len(copies) == 3
  viewer._scene._scene_offset = np.array([1.0, 0.0, 0.0])
  assert camera.needs_update_for(0, viewer._scene._scene_offset)
  viewer._update_camera_feeds(sim, has_pending_updates=False)
  assert not camera.needs_update_for(0, viewer._scene._scene_offset)
  assert [image.writes for image in images] == [1, 2, 1]
  assert len(copies) == 3
  viewer._scene.env_idx = 1
  viewer._update_camera_feeds(sim, has_pending_updates=False)
  assert [image.writes for image in images] == [2, 3, 2]
  assert len(copies) == 6


def test_camera_generation_tracks_rendered_buffers(
  camera_viewer, monkeypatch: pytest.MonkeyPatch
):
  viewer, sensor, _images, _controls, _copies, _data = camera_viewer
  initial = sensor.frame_id
  sensor.update(0.01)
  sensor.reset()
  assert sensor.frame_id == initial
  context = SensorContext.__new__(SensorContext)
  context.camera_sensors = [sensor]
  context.raycast_sensors = []
  context.finalize()
  assert sensor.frame_id == initial + 1
  context.finalize()
  assert sensor.frame_id == initial + 2
  _ = sensor.data
  assert sensor._cache_valid
  monkeypatch.setattr(context, "_create_context", MagicMock())
  context.recreate(viewer._mj_model)
  assert sensor.frame_id == initial + 3
  assert not sensor._cache_valid
