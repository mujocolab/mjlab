"""Viser play viewer extended with a live "Robot Health" tab.

Subclasses ``ViserPlayViewer`` rather than modifying it: the base class
has no plugin hook to add a tab after ``setup()`` runs (its ``tabs``
tab-group is a local variable, not stored on ``self``), but ``setup()``,
``_update_env_dependent_plots()``, ``reset_environment()``, and
``close()`` are all plain overridable methods -- the same seams
``ViserTermOverlays`` (mjlab's own Rewards/Metrics tabs) is wired through
internally. A second ``add_tab_group()`` call produces a second, visually
stacked tab bar below the existing one, since there's no way to inject
into the first tab bar without forking ``setup()`` wholesale.
"""

from __future__ import annotations

import viser

from heat_bench.viewer.health_overlay import RobotHealthOverlay
from mjlab.viewer.viser.viewer import ViserPlayViewer


class HealthMonitoringViewer(ViserPlayViewer):
  """``ViserPlayViewer`` plus a "Robot Health" tab with per-node telemetry."""

  def setup(self) -> None:
    super().setup()

    term = self.env.unwrapped.observation_manager.get_term_cfg(
      "thermal", "thermal_energy"
    ).func
    self._health_overlay = RobotHealthOverlay(self._server, term)

    health_tabs = self._server.gui.add_tab_group()
    with health_tabs.add_tab("Robot Health", icon=viser.Icon.ACTIVITY_HEARTBEAT):
      self._health_overlay.setup_tab()

    self._last_health_env_idx = self._scene.env_idx

  def _update_env_dependent_plots(self) -> None:
    super()._update_env_dependent_plots()
    if self._scene.env_idx != self._last_health_env_idx:
      self._health_overlay.on_env_switch()
      self._last_health_env_idx = self._scene.env_idx
    self._health_overlay.update(self._is_paused, self._scene.env_idx)

  def reset_environment(self) -> None:
    super().reset_environment()
    self._health_overlay.clear_histories()

  def close(self) -> None:
    self._health_overlay.cleanup()
    super().close()
