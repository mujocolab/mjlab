"""Robot Health overlay: live per-node thermal/electrical/torque monitoring.

Mirrors ``ViserTermOverlays`` (``src/mjlab/viewer/viser/overlays.py``) in
shape -- a plain class owning GUI widget handles and per-node history
buffers -- but plots whole vectors (14 thermal nodes, 12 joints) on
shared-axis multi-series charts instead of one scalar-per-term widget,
since the generic Metrics tab can't group nodes together (see the plan
this implements for why). This overlay is read-only visualization: it
does not flag, threshold, or classify anything.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import viser
import viser.uplot

from heat_bench.viewer.node_bar_panel import NodeBarPanel
from mjlab.viewer.viser.term_plotter import _color_for

HISTORY_LENGTH = 300


@dataclass
class _MultiSeriesChart:
  """One shared-axis uplot with one line per node."""

  node_names: list[str]
  history: list[deque[float]] = field(init=False)
  plot: viser.GuiUplotHandle | None = field(default=None, init=False)
  _x: np.ndarray = field(init=False)

  def __post_init__(self) -> None:
    self.history = [deque(maxlen=HISTORY_LENGTH) for _ in self.node_names]
    self._x = np.arange(-HISTORY_LENGTH, 0, dtype=np.float64)

  def create(self, server: viser.ViserServer, title: str) -> None:
    empty = np.array([], dtype=np.float64)
    series = [viser.uplot.Series(label="Steps")]
    for i, name in enumerate(self.node_names):
      series.append(viser.uplot.Series(label=name, stroke=_color_for(i), width=1.5))
    data = (empty,) + tuple(empty for _ in self.node_names)
    self.plot = server.gui.add_uplot(
      data=data,
      series=tuple(series),
      scales={
        "x": viser.uplot.Scale(time=False, auto=False, range=(-HISTORY_LENGTH, 0)),
        "y": viser.uplot.Scale(auto=True),
      },
      legend=viser.uplot.Legend(show=True),
      title=title,
      aspect=2.5,
      visible=True,
    )

  def push(self, values: np.ndarray) -> None:
    for h, v in zip(self.history, values, strict=True):
      if np.isfinite(v):
        h.append(float(v))
    if self.plot is None:
      return
    hist_len = len(self.history[0])
    if hist_len == 0:
      return
    x = self._x[-hist_len:]
    ys = tuple(np.fromiter(h, dtype=np.float64, count=len(h)) for h in self.history)
    self.plot.data = (x,) + ys

  def clear(self) -> None:
    for h in self.history:
      h.clear()

  def cleanup(self) -> None:
    if self.plot is not None:
      self.plot.remove()
      self.plot = None


class RobotHealthOverlay:
  """Owns the Robot Health tab's widgets and per-frame updates."""

  def __init__(self, server: viser.ViserServer, term: Any) -> None:
    """Args:
    server: The Viser server instance.
    term: The live ``ThermalEnergyObservation`` instance (resolved via
      ``env.unwrapped.observation_manager.get_term_cfg(...).func``).
    """
    self._server = server
    self._term = term

    # `term.joint_names` are the actual resolved joint names (e.g.
    # "FR_hip_joint"), in the same order as last_current/last_torque and
    # thermal.T's first 12 entries -- see ThermalEnergyObservation.__init__.
    self.joint_names: list[str] = list(term.joint_names)
    self.thermal_node_names: list[str] = self.joint_names + ["chassis", "ambient"]

    self._thermal_chart = _MultiSeriesChart(self.thermal_node_names)
    self._current_chart = _MultiSeriesChart(self.joint_names)
    self._torque_chart = _MultiSeriesChart(self.joint_names)

    self._thermal_bars: NodeBarPanel | None = None
    self._current_bars: NodeBarPanel | None = None
    self._torque_bars: NodeBarPanel | None = None
    self._soc_bar = None
    self._soc_readout: viser.GuiNumberHandle | None = None
    self._bus_voltage_chart = _MultiSeriesChart(["bus_voltage_v"])
    self._cum_energy_readout: viser.GuiNumberHandle | None = None

  def setup_tab(self) -> None:
    """Build all widgets. Call from inside a ``with tabs.add_tab(...):`` block."""
    self._server.gui.add_markdown("### Thermal (14 nodes)")
    self._thermal_chart.create(self._server, "Joint/chassis/ambient temperature (C)")
    self._thermal_bars = NodeBarPanel(
      self._server, self.thermal_node_names, low=0.0, high=80.0, unit="C"
    )

    self._server.gui.add_markdown("### Current (12 joints)")
    self._current_chart.create(self._server, "Per-actuator current (A)")
    self._current_bars = NodeBarPanel(
      self._server, self.joint_names, low=0.0, high=20.0, unit="A"
    )

    self._server.gui.add_markdown("### Torque (12 joints)")
    self._torque_chart.create(self._server, "Per-actuator torque (N*m)")
    self._torque_bars = NodeBarPanel(
      self._server, self.joint_names, low=0.0, high=35.0, unit="Nm"
    )

    self._server.gui.add_markdown("### Battery")
    # Initialize from the live SoC (whatever `initial_soc` the config sets)
    # rather than a hardcoded 100.0, so it's correct even before the first
    # per-frame update() call runs.
    initial_soc_pct = float(self._term.battery.soc[0].item()) * 100.0
    self._soc_bar = self._server.gui.add_progress_bar(value=initial_soc_pct)
    self._soc_readout = self._server.gui.add_number(
      "SoC (%)", initial_value=initial_soc_pct, disabled=True
    )
    self._bus_voltage_chart.create(self._server, "Bus voltage (V)")
    self._cum_energy_readout = self._server.gui.add_number(
      "Cumulative energy (Wh)", initial_value=0.0, disabled=True
    )

  def update(self, paused: bool, env_idx: int) -> None:
    if paused:
      return
    term = self._term
    temps = term.thermal.T[env_idx, :14].detach().cpu().numpy()
    current = term.last_current[env_idx].detach().cpu().numpy()
    torque = term.last_torque[env_idx].detach().cpu().numpy()

    self._thermal_chart.push(temps)
    self._current_chart.push(current)
    self._torque_chart.push(torque)

    if self._thermal_bars is not None:
      self._thermal_bars.update(temps)
    if self._current_bars is not None:
      self._current_bars.update(current)
    if self._torque_bars is not None:
      self._torque_bars.update(torque)

    soc_pct = float(term.battery.soc[env_idx].item()) * 100.0
    if self._soc_bar is not None:
      self._soc_bar.value = soc_pct
    if self._soc_readout is not None:
      self._soc_readout.value = soc_pct
    self._bus_voltage_chart.push(
      np.array([float(term.battery.bus_voltage[env_idx].item())])
    )
    if self._cum_energy_readout is not None:
      self._cum_energy_readout.value = float(term.battery.cum_wh[env_idx].item())

  def on_env_switch(self) -> None:
    self.clear_histories()

  def clear_histories(self) -> None:
    self._thermal_chart.clear()
    self._current_chart.clear()
    self._torque_chart.clear()
    self._bus_voltage_chart.clear()
    if self._thermal_bars is not None:
      self._thermal_bars.clear()
    if self._current_bars is not None:
      self._current_bars.clear()
    if self._torque_bars is not None:
      self._torque_bars.clear()

  def cleanup(self) -> None:
    self._thermal_chart.cleanup()
    self._current_chart.cleanup()
    self._torque_chart.cleanup()
    self._bus_voltage_chart.cleanup()
    if self._thermal_bars is not None:
      self._thermal_bars.cleanup()
    if self._current_bars is not None:
      self._current_bars.cleanup()
    if self._torque_bars is not None:
      self._torque_bars.cleanup()
