"""mjlab play viewer based on Viser with simulation controls.

Adapted from an MJX visualizer by Chung Min Kim: https://github.com/chungmin99/
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from threading import Lock

import viser
from typing_extensions import override

from mjlab.sim.sim import Simulation
from mjlab.viewer.base import BaseViewer, EnvProtocol, PolicyProtocol, VerbosityLevel
from mjlab.viewer.viser.overlays import (
  ViserCameraOverlays,
  ViserContactOverlays,
  ViserDebugOverlays,
  ViserTermOverlays,
)
from mjlab.viewer.viser.scene import ViserMujocoScene


class UpdateReason(Enum):
  ACTION = auto()
  ENV_SWITCH = auto()
  SCENE_REQUEST = auto()


class ViserPlayViewer(BaseViewer):
  """Interactive Viser-based viewer with playback controls."""

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
  ) -> None:
    super().__init__(env, policy, frame_rate, verbosity)
    self._term_overlays: ViserTermOverlays | None = None
    self._camera_overlays: ViserCameraOverlays | None = None
    self._debug_overlays: ViserDebugOverlays | None = None
    self._contact_overlays: ViserContactOverlays | None = None
    self._sim_lock = Lock()
    self._camera_update_last_ms: float = 0.0
    self._debug_queue_last_ms: float = 0.0
    self._scene_submit_enqueue_last_ms: float = 0.0
    self._scene_update_last_ms: float = 0.0
    self._timing_last_log_time: float = 0.0

  @override
  def setup(self) -> None:
    """Setup the viewer resources."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)

    self._server = viser.ViserServer(label="mjlab")
    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._counter = 0
    self._pending_update_reasons: set[UpdateReason] = set()

    # Create ViserMujocoScene for all 3D visualization (with debug visualization enabled).
    self._scene = ViserMujocoScene.create(
      server=self._server,
      mj_model=sim.mj_model,
      num_envs=self.env.num_envs,
    )

    self._scene.env_idx = self.cfg.env_idx
    self._scene.debug_visualization_enabled = (
      True  # Enable debug visualization by default
    )

    # Create tab group.
    tabs = self._server.gui.add_tab_group()

    # Main tab with simulation controls and display settings.
    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
      # Status display.
      with self._server.gui.add_folder("Info"):
        self._status_html = self._server.gui.add_html("")

      # Simulation controls.
      with self._server.gui.add_folder("Simulation"):
        # Play/Pause button.
        self._pause_button = self._server.gui.add_button(
          "Play" if self._is_paused else "Pause",
          icon=viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE,
        )

        @self._pause_button.on_click
        def _(_) -> None:
          self.request_toggle_pause()

        # Single-step button.
        self._step_button = self._server.gui.add_button(
          "Step",
          icon=viser.Icon.PLAYER_TRACK_NEXT,
        )

        @self._step_button.on_click
        def _(_) -> None:
          self.request_single_step()

        # Reset button.
        reset_button = self._server.gui.add_button("Reset Environment")

        @reset_button.on_click
        def _(_) -> None:
          self.request_reset()

        # Speed controls.
        speed_buttons = self._server.gui.add_button_group(
          "Speed",
          options=["Slower", "1x", "Faster"],
        )

        @speed_buttons.on_click
        def _(event) -> None:
          if event.target.value == "Slower":
            self.request_speed_down()
          elif event.target.value == "1x":
            self.request_reset_speed()
          else:
            self.request_speed_up()

      self._camera_overlays = ViserCameraOverlays(self._server, self.env, sim.mj_model)
      with self._server.gui.add_folder("Camera Feeds"):
        self._camera_overlays.setup_controls()

      # Add standard visualization options from ViserMujocoScene (Environment, Visualization, Contacts, Camera Tracking, Debug Visualization).
      self._scene.create_visualization_gui(
        camera_distance=self.cfg.distance,
        camera_azimuth=self.cfg.azimuth,
        camera_elevation=self.cfg.elevation,
      )

    self._prev_env_idx = self._scene.env_idx

    self._term_overlays = ViserTermOverlays(self._server, self.env, self._scene)
    self._term_overlays.setup_tabs(tabs)
    self._debug_overlays = ViserDebugOverlays(self.env, self._scene)
    self._contact_overlays = ViserContactOverlays(self._scene)

    # Groups tab (geoms and sites).
    self._scene.create_groups_gui(tabs)

  @override
  def _process_actions(self) -> None:
    """Process queued actions and sync UI state."""
    had_actions = bool(self._actions)
    super()._process_actions()
    if had_actions:
      self._pending_update_reasons.add(UpdateReason.ACTION)
      self._sync_ui_state()

  def _sync_ui_state(self) -> None:
    """Sync UI elements to current state after action processing."""
    self._pause_button.label = "Play" if self._is_paused else "Pause"
    self._pause_button.icon = (
      viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE
    )
    self._update_status_display()

  def _update_env_dependent_plots(self) -> None:
    """Refresh reward/metric plots and histories for the selected environment."""
    if self._scene.env_idx != self._prev_env_idx:
      self._prev_env_idx = self._scene.env_idx
      self._pending_update_reasons.add(UpdateReason.ENV_SWITCH)
      if self._term_overlays:
        self._term_overlays.on_env_switch()
      if self._debug_overlays:
        self._debug_overlays.on_env_switch()
      if self._contact_overlays:
        self._contact_overlays.on_env_switch()

    if self._term_overlays:
      self._term_overlays.update(self._is_paused)

  def _update_camera_feeds(self, sim: Simulation, has_pending_updates: bool) -> None:
    """Push camera sensor frames to GUI when needed."""
    t0 = time.perf_counter()
    if self._camera_overlays and self._should_update_cameras(
      self._is_paused, has_pending_updates
    ):
      self._camera_overlays.update(
        sim.data, self._scene.env_idx, self._scene._scene_offset
      )
    self._camera_update_last_ms = (time.perf_counter() - t0) * 1000.0

  def _queue_debug_visualizers(self) -> None:
    """Queue environment-specific debug draw calls into the scene."""
    t0 = time.perf_counter()
    if self._debug_overlays:
      self._debug_overlays.queue()
    self._debug_queue_last_ms = (time.perf_counter() - t0) * 1000.0

  def _submit_scene_update_if_needed(
    self, sim: Simulation, has_pending_updates: bool
  ) -> None:
    """Submit a scene sync job when the update policy allows it."""
    t_enqueue_start = time.perf_counter()
    if self._scene.needs_update:
      self._pending_update_reasons.add(UpdateReason.SCENE_REQUEST)

    if not self._should_submit_scene_update(
      self._counter, self._is_paused, has_pending_updates
    ):
      self._scene_submit_enqueue_last_ms = 0.0
      return

    def update_scene() -> None:
      with self._sim_lock:
        t0 = time.perf_counter()
        with self._server.atomic():
          self._scene.update(sim.data)
          self._server.flush()
        self._scene_update_last_ms = (time.perf_counter() - t0) * 1000.0

    self._threadpool.submit(update_scene)
    self._scene_submit_enqueue_last_ms = (
      time.perf_counter() - t_enqueue_start
    ) * 1000.0
    self._pending_update_reasons.clear()
    self._scene.needs_update = False

  def _maybe_log_debug_timings(self) -> None:
    """Log lightweight Viser pipeline timing in debug mode."""
    if self.verbosity < VerbosityLevel.DEBUG:
      return
    now = time.time()
    if now - self._timing_last_log_time < 1.0:
      return
    self._timing_last_log_time = now
    self.log(
      (
        "[DEBUG] Viser timings: "
        f"camera={self._camera_update_last_ms:.2f}ms, "
        f"debug={self._debug_queue_last_ms:.2f}ms, "
        f"submit_enqueue={self._scene_submit_enqueue_last_ms:.2f}ms, "
        f"scene_update={self._scene_update_last_ms:.2f}ms"
      ),
      VerbosityLevel.DEBUG,
    )

  @staticmethod
  def _should_update_cameras(paused: bool, has_pending_updates: bool) -> bool:
    """Camera feeds update continuously while running and on-demand while paused."""
    return (not paused) or has_pending_updates

  @staticmethod
  def _should_submit_scene_update(
    counter: int, paused: bool, has_pending_updates: bool
  ) -> bool:
    """Scene submits at 30Hz (every other 60Hz tick) with pause-aware gating."""
    if counter % 2 != 0:
      return False
    if paused and not has_pending_updates:
      return False
    return True

  @override
  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    self._scene.paused = self._is_paused
    self._counter += 1
    if self._counter % 10 == 0:
      self._update_status_display()
    self._update_env_dependent_plots()
    has_pending_updates = bool(self._pending_update_reasons) or self._scene.needs_update
    self._update_camera_feeds(sim, has_pending_updates)
    self._queue_debug_visualizers()
    self._submit_scene_update_if_needed(sim, has_pending_updates)
    self._maybe_log_debug_timings()

  @override
  def sync_viewer_to_env(self) -> None:
    """Synchronize viewer state to environment (e.g., perturbations)."""
    pass

  @override
  def reset_environment(self) -> None:
    """Extend BaseViewer.reset_environment to clear reward and metrics histories."""
    with self._sim_lock:
      super().reset_environment()
    if self._term_overlays:
      self._term_overlays.clear_histories()

  @override
  def close(self) -> None:
    """Close the viewer and cleanup resources."""
    if self._term_overlays:
      self._term_overlays.cleanup()
    if self._camera_overlays:
      self._camera_overlays.cleanup()
    self._threadpool.shutdown(wait=True)
    self._server.stop()

  @override
  def is_running(self) -> bool:
    """Check if viewer is running."""
    return True  # Viser runs until process is killed.

  def _update_status_display(self) -> None:
    """Update the HTML status display."""
    status = self.get_status()
    actual_rt = status.actual_realtime
    rt_display = f"{actual_rt:.2f}x" if actual_rt > 0 else "—"
    capped = ' <span style="color:#e74c3c;">[CAPPED]</span>' if status.capped else ""
    error_line = ""
    if status.last_error:
      # Show last line of traceback to avoid flooding the panel.
      first_line = status.last_error.strip().splitlines()[-1]
      error_line = (
        f'<br/><span style="color:#e74c3c;"><strong>Error:</strong> {first_line}</span>'
      )
    self._status_html.content = f"""
      <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
        <strong>Status:</strong> {"Paused" if status.paused else "Running"}{capped}<br/>
        <strong>Steps:</strong> {status.step_count}<br/>
        <strong>Speed:</strong> {status.speed_label}<br/>
        <strong>Target RT:</strong> {status.target_realtime:.2f}x<br/>
        <strong>Actual RT:</strong> {rt_display} ({status.smoothed_fps:.0f} FPS){error_line}
      </div>
      """
