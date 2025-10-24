"""Mjlab play viewer based on Viser with simulation controls.

Adapted from an MJX visualizer by Chung Min Kim: https://github.com/chungmin99/
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import viser
from typing_extensions import override

from mjlab.entity.entity import Entity
from mjlab.sim.sim import Simulation
from mjlab.viewer.base import BaseViewer, EnvProtocol, PolicyProtocol, VerbosityLevel
from mjlab.viewer.viser_reward_plotter import ViserRewardPlotter
from mjlab.viewer.viser_scene import ViserMujocoScene
from mjlab.viewer.viser_visualizer import ViserDebugVisualizer


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
    self._reward_plotter: Optional[ViserRewardPlotter] = None
    self._debug_visualizer: Optional[ViserDebugVisualizer] = None

  @override
  def setup(self) -> None:
    """Setup the viewer resources."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)

    self._server = viser.ViserServer(label="mjlab")
    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._counter = 0
    self._show_debug_vis = True
    self._camera_tracking = False
    self._needs_update = False

    # Create ViserMujocoScene for all 3D visualization.
    self._scene = ViserMujocoScene.create(
      server=self._server,
      mj_model=sim.mj_model,
      num_envs=self.env.num_envs,
    )

    # Set initial environment index from config.
    self._scene.current_env_idx = self.cfg.env_idx

    # Setup camera tracking body if configured.
    if self.cfg and self.cfg.asset_name and self.cfg.body_name:
      robot: Entity = self.env.unwrapped.scene[self.cfg.asset_name]
      if self.cfg.body_name not in robot.body_names:
        raise ValueError(
          f"Body '{self.cfg.body_name}' not found in asset '{self.cfg.asset_name}'"
        )
      body_indices_and_names = robot.find_bodies(self.cfg.body_name)
      body_id_list = body_indices_and_names[0]
      self._tracked_body_id = robot.indexing.bodies[body_id_list[0]].id
    else:
      # Fallback: use body 1 (first body after world).
      self._tracked_body_id = 1

    # Create tabs.
    tabs = self._server.gui.add_folder("Controls")

    # Main tab with simulation controls and display settings.
    with tabs:
      # Status display.
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
          self.toggle_pause()
          self._pause_button.label = "Play" if self._is_paused else "Pause"
          self._pause_button.icon = (
            viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE
          )
          self._update_status_display()
          self._needs_update = True

        # Reset button.
        reset_button = self._server.gui.add_button("Reset Environment")

        @reset_button.on_click
        def _(_) -> None:
          self.reset_environment()
          self._update_status_display()
          self._needs_update = True

        # Speed controls.
        speed_buttons = self._server.gui.add_button_group(
          "Speed",
          options=["Slower", "Faster"],
        )

        @speed_buttons.on_click
        def _(event) -> None:
          if event.target.value == "Slower":
            self.decrease_speed()
          else:
            self.increase_speed()
          self._update_status_display()

      # Camera tracking controls.
      with self._server.gui.add_folder("Camera"):
        cb_camera_tracking = self._server.gui.add_checkbox(
          "Track",
          initial_value=False,
          hint="Keep tracked body centered. Use Viser camera controls to adjust view.",
        )

        @cb_camera_tracking.on_update
        def _(_) -> None:
          self._camera_tracking = cb_camera_tracking.value
          self._needs_update = True
          # When enabling tracking, set all camera look-ats and positions to config defaults.
          if self._camera_tracking:
            # Get camera parameters from config.
            distance = self.cfg.distance
            azimuth = self.cfg.azimuth
            elevation = self.cfg.elevation

            # Convert to radians and calculate camera position.
            azimuth_rad = np.deg2rad(azimuth)
            elevation_rad = np.deg2rad(elevation)

            # Calculate forward vector from spherical coordinates.
            forward = np.array(
              [
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad),
              ]
            )

            # Camera position is origin - forward * distance.
            camera_pos = -forward * distance

            for client in self._server.get_clients().values():
              client.camera.position = camera_pos
              client.camera.look_at = np.zeros(3)

      # Debug visualization controls.
      with self._server.gui.add_folder("Debug"):
        cb_debug_vis = self._server.gui.add_checkbox(
          "Show",
          initial_value=True,
          hint="Show debug arrows and ghost meshes.",
        )

        @cb_debug_vis.on_update
        def _(_) -> None:
          self._show_debug_vis = cb_debug_vis.value
          self._needs_update = True
          # Clear visualizer if hiding.
          if not self._show_debug_vis and self._debug_visualizer is not None:
            self._debug_visualizer.clear_all()

    # Add standard visualization options from ViserMujocoScene.
    self._scene.create_options_gui()

    # Store previous env idx to detect changes (for clearing reward histories).
    self._prev_env_idx = self._scene.current_env_idx

    # Setup environment slider callback to clear reward histories on change.
    # We do this after create_options_gui() so we can access the slider GUI state.
    if self.env.num_envs > 1:
      # Note: ViserScene already created the slider, we just need to react to changes.
      pass

    # Reward plots tab.
    if hasattr(self.env.unwrapped, "reward_manager"):
      with self._server.gui.add_folder("Rewards"):
        # Get reward term names and create reward plotter.
        term_names = [
          name
          for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self._scene.current_env_idx
          )
        ]
        self._reward_plotter = ViserRewardPlotter(self._server, term_names)

  @override
  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)

    # Update counter.
    self._counter += 1

    # Update status display and reward plots less frequently.
    if self._counter % 10 == 0:
      self._update_status_display()

      # Check if environment changed.
      if self._scene.current_env_idx != self._prev_env_idx:
        self._prev_env_idx = self._scene.current_env_idx
        if self._reward_plotter:
          self._reward_plotter.clear_histories()

      if self._reward_plotter is not None and not self._is_paused:
        terms = list(
          self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self._scene.current_env_idx
          )
        )
        self._reward_plotter.update(terms)

    # Enable camera tracking if requested.
    if self._camera_tracking:
      self._scene.camera_tracking_body_id = self._tracked_body_id
    else:
      self._scene.camera_tracking_body_id = None

    # Compute scene offset for debug visualizer.
    scene_offset = np.zeros(3)
    if self._camera_tracking:
      tracked_pos = sim.wp_data.xpos.numpy()[
        self._scene.current_env_idx, self._tracked_body_id, :
      ].copy()
      scene_offset = -tracked_pos

    # Update debug visualizations every frame.
    if self._show_debug_vis and hasattr(self.env.unwrapped, "update_visualizers"):
      # Only recreate if environment changed or doesn't exist.
      if (
        self._debug_visualizer is None
        or self._debug_visualizer.env_idx != self._scene.current_env_idx
      ):
        # Clear old visualizer completely when switching envs.
        if self._debug_visualizer:
          self._debug_visualizer.clear_all()

        self._debug_visualizer = ViserDebugVisualizer(
          self._server,
          sim.mj_model,
          self._scene.current_env_idx,
          scene_offset,
        )
      else:
        # Just clear arrows and reuse existing visualizer.
        # Ghost meshes kept and poses updated for efficiency.
        self._debug_visualizer.clear()
        self._debug_visualizer.env_origin = scene_offset

      # Update visualizations (queues arrows and updates ghost poses).
      self.env.unwrapped.update_visualizers(self._debug_visualizer)

      # Synchronize queued arrows to the scene.
      self._debug_visualizer._sync_arrows()
    elif not self._show_debug_vis and self._debug_visualizer is not None:
      # Clear visualizer if debug vis is disabled.
      self._debug_visualizer.clear_all()

    # The rest of this method is environment state syncing.
    # It's fine to do this every other policy step to reduce bandwidth usage.
    if self._counter % 2 != 0:
      return

    # Skip scene updates when paused unless UI interaction requires it.
    if self._is_paused and not self._needs_update and not self._scene.needs_update:
      return

    # Update scene asynchronously.
    wp_data = sim.wp_data

    def update_scene() -> None:
      with self._server.atomic():
        # ViserScene handles all mesh and contact visualization.
        self._scene.update(wp_data)
        self._server.flush()

    self._threadpool.submit(update_scene)

    # Clear update flags after syncing.
    self._needs_update = False
    self._scene.needs_update = False

  @override
  def sync_viewer_to_env(self) -> None:
    """Synchronize viewer state to environment (e.g., perturbations)."""
    # Does nothing for Viser.
    pass

  def reset_environment(self) -> None:
    """Extend BaseViewer.reset_environment to clear reward histories."""
    super().reset_environment()
    if self._reward_plotter:
      self._reward_plotter.clear_histories()

  @override
  def close(self) -> None:
    """Close the viewer and cleanup resources."""
    if self._reward_plotter:
      self._reward_plotter.cleanup()
    self._threadpool.shutdown(wait=True)
    self._server.stop()

  @override
  def is_running(self) -> bool:
    """Check if viewer is running."""
    return True  # Viser runs until process is killed.

  def _update_status_display(self) -> None:
    """Update the HTML status display."""
    self._status_html.content = f"""
      <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
        <strong>Status:</strong> {"Paused" if self._is_paused else "Running"}<br/>
        <strong>Steps:</strong> {self._step_count}<br/>
        <strong>Speed:</strong> {self._time_multiplier:.0%}
      </div>
      """
