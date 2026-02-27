"""Tests for Viser viewer update-policy helpers."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from mjlab.viewer.viser.overlays import ViserContactOverlays, ViserDebugOverlays
from mjlab.viewer.viser.scene import ViserMujocoScene
from mjlab.viewer.viser.viewer import ViserPlayViewer


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


def test_scene_requires_live_refresh():
  assert not ViserMujocoScene._requires_live_refresh(
    show_contact_points=False,
    show_contact_forces=False,
    debug_visualization_enabled=False,
  )
  assert ViserMujocoScene._requires_live_refresh(
    show_contact_points=True,
    show_contact_forces=False,
    debug_visualization_enabled=False,
  )
  assert ViserMujocoScene._requires_live_refresh(
    show_contact_points=False,
    show_contact_forces=False,
    debug_visualization_enabled=True,
  )


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
