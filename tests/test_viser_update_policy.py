"""Tests for Viser viewer update-policy helpers."""

from mjlab.viewer.viser.scene import ViserMujocoScene
from mjlab.viewer.viser.viewer import ViserPlayViewer


def test_should_update_cameras():
  assert ViserPlayViewer._should_update_cameras(paused=False, has_pending_updates=False)
  assert ViserPlayViewer._should_update_cameras(paused=False, has_pending_updates=True)
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
    show_contact_forces=True,
    debug_visualization_enabled=False,
  )
  assert ViserMujocoScene._requires_live_refresh(
    show_contact_points=False,
    show_contact_forces=False,
    debug_visualization_enabled=True,
  )
