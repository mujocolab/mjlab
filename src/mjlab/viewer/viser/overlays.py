"""Overlay managers for Viser viewer orchestration.

These managers intentionally coordinate *when* higher-level updates happen
(env switches, paused/running updates, etc.) while leaving low-level render
handle lifecycle ownership inside :mod:`scene.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import mujoco
import viser

from mjlab.sensor import CameraSensor
from mjlab.viewer.viser.camera_viewer import ViserCameraViewer
from mjlab.viewer.viser.term_plotter import ViserTermPlotter


class _EnvProtocol(Protocol):
  @property
  def unwrapped(self) -> Any: ...


class _SceneProtocol(Protocol):
  env_idx: int
  debug_visualization_enabled: bool
  show_contact_points: bool
  show_contact_forces: bool
  needs_update: bool

  def clear_debug_all(self) -> None: ...
  def clear(self) -> None: ...


@dataclass
class ViserTermOverlays:
  """Manage reward/metrics term plot tabs for Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol
  reward_plotter: ViserTermPlotter | None = None
  metrics_plotter: ViserTermPlotter | None = None

  def setup_tabs(self, tabs: Any) -> None:
    """Create rewards/metrics tabs based on available managers."""
    if hasattr(self.env.unwrapped, "reward_manager"):
      with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
        term_names = [
          name
          for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self.scene.env_idx
          )
        ]
        self.reward_plotter = ViserTermPlotter(self.server, term_names, name="Reward")

    if hasattr(self.env.unwrapped, "metrics_manager"):
      term_names = [
        name
        for name, _ in self.env.unwrapped.metrics_manager.get_active_iterable_terms(
          self.scene.env_idx
        )
      ]
      if term_names:
        with tabs.add_tab("Metrics", icon=viser.Icon.CHART_BAR):
          self.metrics_plotter = ViserTermPlotter(
            self.server, term_names, name="Metric"
          )

  def on_env_switch(self) -> None:
    """Clear histories when active environment changes."""
    if self.reward_plotter:
      self.reward_plotter.clear_histories()
    if self.metrics_plotter:
      self.metrics_plotter.clear_histories()

  def update(self, paused: bool) -> None:
    """Update term plots from the selected environment."""
    if self.reward_plotter is not None and not paused:
      terms = list(
        self.env.unwrapped.reward_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      self.reward_plotter.update(terms)

    if self.metrics_plotter is not None and not paused:
      terms = list(
        self.env.unwrapped.metrics_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      self.metrics_plotter.update(terms)

  def clear_histories(self) -> None:
    """Clear all overlay histories."""
    self.on_env_switch()

  def cleanup(self) -> None:
    """Cleanup plotter resources."""
    if self.reward_plotter:
      self.reward_plotter.cleanup()
    if self.metrics_plotter:
      self.metrics_plotter.cleanup()


@dataclass
class ViserCameraOverlays:
  """Manage camera feed widgets and updates for Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  mj_model: mujoco.MjModel
  camera_viewers: list[ViserCameraViewer] | None = None

  def setup_controls(self) -> None:
    """Create camera feed controls under the active GUI folder."""
    camera_sensors = [
      sensor
      for sensor in self.env.unwrapped.scene.sensors.values()
      if isinstance(sensor, CameraSensor)
    ]
    if not camera_sensors:
      self.camera_viewers = []
      return

    self.camera_viewers = [
      ViserCameraViewer(self.server, sensor, self.mj_model) for sensor in camera_sensors
    ]

  def update(self, sim_data: Any, env_idx: int, scene_offset: Any) -> None:
    """Push latest camera images/frustums to GUI."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.update(sim_data, env_idx, scene_offset)

  def cleanup(self) -> None:
    """Cleanup all camera feed widgets."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.cleanup()


@dataclass
class ViserDebugOverlays:
  """Manage debug visualization queueing and env-switch behavior."""

  env: _EnvProtocol
  scene: _SceneProtocol

  def on_env_switch(self) -> None:
    """Reset debug visuals when switching selected environment."""
    if self.scene.debug_visualization_enabled:
      self.scene.clear_debug_all()

  def queue(self) -> None:
    """Queue environment debug visualizers for the current frame."""
    if self.scene.debug_visualization_enabled and hasattr(
      self.env.unwrapped, "update_visualizers"
    ):
      self.scene.clear()  # Clear queued arrows from previous frame.
      self.env.unwrapped.update_visualizers(self.scene)


@dataclass
class ViserContactOverlays:
  """Manage contact-visualization orchestration from the viewer layer.

  Note: contact mesh creation/update/removal stays in ``ViserMujocoScene``.
  This manager only requests scene refreshes at the right times.
  """

  scene: _SceneProtocol

  def is_enabled(self) -> bool:
    """Whether any contact visualization is currently enabled."""
    return self.scene.show_contact_points or self.scene.show_contact_forces

  def on_env_switch(self) -> None:
    """Request a scene refresh when switching environments with contacts enabled."""
    if self.is_enabled():
      self.scene.needs_update = True
