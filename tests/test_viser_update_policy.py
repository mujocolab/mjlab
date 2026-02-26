"""Tests for Viser viewer update-policy helpers."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from mjlab.viewer.viser.scene import ViserMujocoScene
from mjlab.viewer.viser.viewer import ViserPlayViewer


@dataclass
class _DummyHandle:
  visible: bool


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


def test_sync_ghosts_empty_queue_paused_visibility():
  scene = object.__new__(ViserMujocoScene)
  scene.debug_visualization_enabled = True
  scene._queued_ghosts = []
  scene.paused = True
  h1 = _DummyHandle(visible=False)
  h2 = _DummyHandle(visible=False)
  scene._ghost_handles_batched = cast(Any, {(1, 1): h1, (2, 2): h2})

  ViserMujocoScene._sync_ghosts(scene)
  assert h1.visible
  assert h2.visible

  scene.paused = False
  ViserMujocoScene._sync_ghosts(scene)
  assert not h1.visible
  assert not h2.visible


def test_refresh_visualization_sets_needs_update_for_live_overlays():
  scene = object.__new__(ViserMujocoScene)
  scene._last_body_xpos = np.zeros((1, 1, 3), dtype=np.float32)
  scene._last_body_xmat = np.zeros((1, 1, 3, 3), dtype=np.float32)
  scene._last_mocap_pos = np.zeros((1, 0, 3), dtype=np.float32)
  scene._last_mocap_quat = np.zeros((1, 0, 4), dtype=np.float32)
  scene._last_env_idx = 0
  scene._tracked_body_id = None
  scene.camera_tracking_enabled = False
  scene._last_contacts = []

  def _noop_update_visualization(
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    contacts: list[Any] | None,
  ) -> None:
    del body_xpos, body_xmat, mocap_pos, mocap_quat, env_idx, scene_offset, contacts

  scene._update_visualization = _noop_update_visualization

  scene.show_contact_points = False
  scene.show_contact_forces = False
  scene.debug_visualization_enabled = False
  ViserMujocoScene.refresh_visualization(scene)
  assert not scene.needs_update

  scene.debug_visualization_enabled = True
  ViserMujocoScene.refresh_visualization(scene)
  assert scene.needs_update
