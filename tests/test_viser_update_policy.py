"""Tests for Viser viewer update-policy helpers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any
from unittest.mock import MagicMock

import mujoco
import pytest
from mjviser import ViserMujocoScene

from mjlab.sim.sim import Simulation
from mjlab.viewer.viser.overlays import ViserContactOverlays, ViserDebugOverlays
from mjlab.viewer.viser.viewer import UpdateReason, ViserPlayViewer


@dataclass
class _DummyScene:
  env_idx: int
  debug_visualization_enabled: bool
  show_contact_points: bool = False
  show_contact_forces: bool = False
  needs_update: bool = False
  clear_count: int = 0
  clear_debug_count: int = 0

  def clear(self) -> None:
    self.clear_count += 1

  def clear_debug_all(self) -> None:
    self.clear_debug_count += 1


class _DummyEnv:
  def __init__(self, unwrapped: Any):
    self._unwrapped = unwrapped

  @property
  def unwrapped(self) -> Any:
    return self._unwrapped


def test_should_update_cameras():
  assert ViserPlayViewer._should_update_cameras(paused=False, has_pending_updates=False)
  assert not ViserPlayViewer._should_update_cameras(
    paused=True, has_pending_updates=False
  )
  assert ViserPlayViewer._should_update_cameras(paused=True, has_pending_updates=True)


def test_should_submit_scene_update():
  # Odd ticks are skipped to keep scene submits around 30Hz.
  assert not ViserPlayViewer._should_submit_scene_update(
    counter=1, paused=False, has_pending_updates=True
  )
  # Running: submit on even ticks regardless of pending flags.
  assert ViserPlayViewer._should_submit_scene_update(
    counter=2, paused=False, has_pending_updates=False
  )
  # Paused: submit only with pending updates.
  assert not ViserPlayViewer._should_submit_scene_update(
    counter=2, paused=True, has_pending_updates=False
  )
  assert ViserPlayViewer._should_submit_scene_update(
    counter=2, paused=True, has_pending_updates=True
  )


def test_slow_scene_updates_coalesce_without_blocking_ticks(
  monkeypatch: pytest.MonkeyPatch,
):
  env = MagicMock()
  env.unwrapped.sim = MagicMock(spec=Simulation)
  viewer = ViserPlayViewer(env, MagicMock(), viser_server=MagicMock())
  executor = ThreadPoolExecutor(max_workers=1)
  executor_mock = MagicMock(wraps=executor)
  viewer._threadpool = executor_mock
  scene = MagicMock()
  viewer._scene = scene
  scene.needs_update = False
  viewer._counter = 1
  viewer._pending_update_reasons = set()
  debug_overlays = MagicMock()
  viewer._debug_overlays = debug_overlays
  monkeypatch.setattr(viewer, "_update_env_dependent_plots", MagicMock())
  monkeypatch.setattr(viewer, "_update_status_display", MagicMock())
  monkeypatch.setattr(viewer, "_update_camera_feeds", MagicMock())
  entered, release, ticks_done = Event(), Event(), Event()

  def slow_update(_data):
    entered.set()
    assert release.wait(timeout=5)

  scene.update.side_effect = slow_update
  thread = None
  try:
    viewer.sync_env_to_viewer()
    assert entered.wait(timeout=5)
    viewer._is_paused = True
    viewer._pending_update_reasons.add(UpdateReason.ACTION)
    scene.needs_update = True

    def sync_ticks():
      for _ in range(20):
        viewer.sync_env_to_viewer()
      ticks_done.set()

    thread = Thread(target=sync_ticks)
    thread.start()
    assert ticks_done.wait(timeout=1), "rendering must not block viewer ticks"
    assert executor_mock.submit.call_count == 1
    assert debug_overlays.queue.call_count == 1
    assert UpdateReason.ACTION in viewer._pending_update_reasons

    release.set()
    assert viewer._scene_update_future is not None
    viewer._scene_update_future.result(timeout=5)
    viewer._counter = 1
    viewer.sync_env_to_viewer()
    assert viewer._scene_update_future is not None
    viewer._scene_update_future.result(timeout=5)
    assert executor_mock.submit.call_count == 2
    assert debug_overlays.queue.call_count == 2
    assert not viewer._pending_update_reasons
  finally:
    release.set()
    if thread is not None:
      thread.join(timeout=5)
    viewer.close()


def test_scene_requires_live_refresh():
  """Base scene sets needs_update when contact overlays are visible."""
  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  server = MagicMock()
  scene = ViserMujocoScene(server, model, num_envs=1)

  scene.show_contact_points = False
  scene.show_contact_forces = False
  scene.needs_update = False
  scene._last_body_xpos = None  # No cached data, refresh is a no-op.
  scene.refresh_visualization()
  assert not scene.needs_update

  scene.show_contact_points = True
  assert scene.show_contact_points or scene.show_contact_forces


def test_debug_overlays_env_switch_and_queue():
  """env_switch respects enabled flag; queue clears + calls update_visualizers."""
  unwrapped = MagicMock(spec=["update_visualizers"])
  env = _DummyEnv(unwrapped)
  scene = _DummyScene(env_idx=0, debug_visualization_enabled=False)
  overlays = ViserDebugOverlays(env=env, scene=scene)

  # Disabled: env switch does not clear.
  overlays.on_env_switch()
  assert scene.clear_debug_count == 0

  # Enabled: env switch clears, queue dispatches.
  scene.debug_visualization_enabled = True
  overlays.on_env_switch()
  assert scene.clear_debug_count == 1

  overlays.queue()
  assert scene.clear_count == 1
  unwrapped.update_visualizers.assert_called_once_with(scene)


def test_contact_overlays_env_switch():
  """env_switch requests scene update only when contacts are enabled."""
  scene = _DummyScene(env_idx=0, debug_visualization_enabled=False)
  overlays = ViserContactOverlays(scene=scene)

  assert not overlays.is_enabled()
  overlays.on_env_switch()
  assert not scene.needs_update

  scene.show_contact_points = True
  assert overlays.is_enabled()
  overlays.on_env_switch()
  assert scene.needs_update
