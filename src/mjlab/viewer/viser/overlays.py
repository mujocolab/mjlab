"""Overlay helpers for Viser viewer tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import viser

from mjlab.viewer.viser.term_plotter import ViserTermPlotter


@dataclass
class ViserTermOverlays:
  """Manage reward/metrics term plot tabs for Viser viewer."""

  server: viser.ViserServer
  env: Any
  scene: Any
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
