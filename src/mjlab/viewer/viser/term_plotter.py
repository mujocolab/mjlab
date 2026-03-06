"""Plotting functionality for Viser viewer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import viser
import viser.uplot

_PALETTE = [
  "#1f77b4",  # blue
  "#ff7f0e",  # orange
  "#2ca02c",  # green
  "#d62728",  # red
  "#9467bd",  # purple
  "#8c564b",  # brown
  "#e377c2",  # pink
  "#7f7f7f",  # gray
  "#bcbd22",  # olive
  "#17becf",  # cyan
  "#aec7e8",  # light blue
  "#ffbb78",  # light orange
]


def _color_for(index: int) -> str:
  return _PALETTE[index % len(_PALETTE)]


@dataclass
class _TermState:
  """Mutable state for a single term."""

  name: str
  color: str
  enabled: bool = False
  history: deque[float] = field(default_factory=lambda: deque(maxlen=300))
  plot: viser.GuiUplotHandle | None = None


class ViserTermPlotter:
  """Handles plotting for the Viser viewer with selective display."""

  def __init__(
    self,
    server: viser.ViserServer,
    term_names: list[str],
    name: str = "Reward",
    history_length: int = 150,
    env_idx: int = 0,
  ) -> None:
    """Initialize the plotter.

    Args:
      server: The Viser server instance
      term_names: List of term names to plot
      name: Name prefix for the plots (e.g. "Reward" or "Metric")
      history_length: Number of points to keep in history
      env_idx: Index of the environment being displayed
    """
    self._server = server
    self._name = name
    self._history_length = history_length

    # Pre-allocated x-axis array (reused for all plots).
    self._x_array = np.arange(-history_length + 1, 1, dtype=np.float64)

    # Stable color assignment.
    self._terms: dict[str, _TermState] = {}
    for i, tname in enumerate(term_names):
      self._terms[tname] = _TermState(
        name=tname,
        color=_color_for(i),
        history=deque(maxlen=history_length),
      )

    # GUI handles.
    self._checkboxes: dict[str, viser.GuiInputHandle] = {}

    self._empty = np.array([], dtype=np.float64)

    self._env_idx = env_idx

    # Build all GUI elements.
    self._build_selector_gui(term_names)
    self._plots_folder = self._server.gui.add_folder("Plots", expand_by_default=True)

  def _build_selector_gui(self, term_names: list[str]) -> None:
    """Build flat checkboxes with a filter input for term selection."""
    with self._server.gui.add_folder("Select terms", expand_by_default=True):
      self._env_label = self._server.gui.add_markdown(self._env_label_text())

      # Filter input.
      self._filter_input = self._server.gui.add_text(
        "Filter",
        initial_value="",
        hint="Term name must contain this string",
      )

      @self._filter_input.on_update
      def _(_) -> None:
        filter_str = self._filter_input.value.lower()
        for tname, state in self._terms.items():
          cb = self._checkboxes[tname]
          visible = filter_str in tname.lower()
          cb.visible = visible
          if not visible:
            if state.plot is not None:
              state.plot.remove()
              state.plot = None
          elif state.enabled and state.plot is None:
            self._create_plot(state)

      # Bulk actions.
      bulk = self._server.gui.add_button_group("Select", options=["All", "None"])

      @bulk.on_click
      def _(event) -> None:
        enable = event.target.value == "All"
        for tname, state in self._terms.items():
          cb = self._checkboxes[tname]
          if cb.visible:
            state.enabled = enable
            cb.value = enable
        self._sync_plots()

      # Flat checkbox list.
      for tname in term_names:
        state = self._terms[tname]
        cb = self._server.gui.add_checkbox(
          tname,
          initial_value=state.enabled,
          hint=f"Color: {state.color}",
        )
        self._checkboxes[tname] = cb

        @cb.on_update
        def _(event, _tname=tname) -> None:
          self._terms[_tname].enabled = event.target.value
          self._sync_plots()

  def _env_label_text(self) -> str:
    return f"<small><em>Showing terms for environment #{self._env_idx}</em></small>"

  def update_env_idx(self, env_idx: int) -> None:
    """Update the displayed environment index."""
    self._env_idx = env_idx
    self._env_label.content = self._env_label_text()

  def _sync_plots(self) -> None:
    """Create or remove plots to match current selection."""
    for tname, state in self._terms.items():
      cb = self._checkboxes[tname]
      should_show = state.enabled and cb.visible
      if should_show and state.plot is None:
        self._create_plot(state)
      elif not should_show and state.plot is not None:
        state.plot.remove()
        state.plot = None

  def _create_plot(self, state: _TermState) -> None:
    """Lazily create a single-term plot inside the scoped folder."""
    h = state.history
    hist_len = len(h)
    if hist_len > 0:
      x = self._x_array[-hist_len:]
      y = np.fromiter(h, dtype=np.float64, count=hist_len)
    else:
      x = self._empty
      y = self._empty

    with self._plots_folder:
      state.plot = self._server.gui.add_uplot(
        data=(x, y),
        series=(
          viser.uplot.Series(label="Steps"),
          viser.uplot.Series(label=state.name, stroke=state.color, width=2),
        ),
        scales={
          "x": viser.uplot.Scale(
            time=False, auto=False, range=(-self._history_length, 0)
          ),
          "y": viser.uplot.Scale(auto=True),
        },
        legend=viser.uplot.Legend(show=False),
        title=state.name,
        aspect=2.0,
        visible=True,
      )

  def update(self, terms: list[tuple[str, np.ndarray]]) -> None:
    """Push new data and refresh visible plots."""
    any_enabled = False
    for tname, arr in terms:
      state = self._terms.get(tname)
      if state is None:
        continue
      val = float(arr[0])
      if np.isfinite(val):
        state.history.append(val)
      if state.enabled:
        any_enabled = True

    if not any_enabled:
      return

    # Update plots.
    for state in self._terms.values():
      if not state.enabled or state.plot is None:
        continue
      h = state.history
      hist_len = len(h)
      if hist_len > 0:
        x = self._x_array[-hist_len:]
        y = np.fromiter(h, dtype=np.float64, count=hist_len)
        state.plot.data = (x, y)

  def clear_histories(self) -> None:
    """Clear all term histories."""
    for state in self._terms.values():
      state.history.clear()
      if state.plot is not None:
        state.plot.data = (self._empty, self._empty)

  def cleanup(self) -> None:
    """Clean up resources."""
    for state in self._terms.values():
      if state.plot is not None:
        state.plot.remove()
    for cb in self._checkboxes.values():
      cb.remove()
    self._terms.clear()
    self._checkboxes.clear()
