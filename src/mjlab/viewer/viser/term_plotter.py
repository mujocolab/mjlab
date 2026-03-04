"""Plotting functionality for Viser viewer."""

from __future__ import annotations

import contextlib
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


def _group_terms(names: list[str], min_group: int = 2) -> dict[str, list[str]]:
  """Group term names by longest common prefix (split on ``_``).

  Terms that don't share a prefix with at least ``min_group - 1`` others
  are placed in an "other" bucket.

  Returns:
    Ordered dict of ``{group_label: [term_names]}``.
  """
  prefix_map: dict[str, list[str]] = {}
  for name in names:
    parts = name.split("_")
    prefix = parts[0] if parts else name
    prefix_map.setdefault(prefix, []).append(name)

  groups: dict[str, list[str]] = {}
  other: list[str] = []
  for prefix, members in sorted(prefix_map.items()):
    if len(members) >= min_group:
      groups[prefix] = sorted(members)
    else:
      other.extend(members)
  if other:
    groups["other"] = sorted(other)
  return groups


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
  ) -> None:
    """Initialize the plotter.

    Args:
      server: The Viser server instance
      term_names: List of term names to plot
      name: Name prefix for the plots (e.g. "Reward" or "Metric")
      history_length: Number of points to keep in history
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

    # Build all GUI elements.
    self._build_selector_gui(term_names)
    self._plots_folder = self._server.gui.add_folder("Plots", expand_by_default=True)

  def _build_selector_gui(self, term_names: list[str]) -> None:
    """Build grouped checkboxes for term selection."""
    with self._server.gui.add_folder("Select terms", expand_by_default=True):
      # Bulk actions.
      bulk = self._server.gui.add_button_group("Bulk", options=["All", "None"])

      @bulk.on_click
      def _(event) -> None:
        enable = event.target.value == "All"
        for tname, state in self._terms.items():
          state.enabled = enable
          self._checkboxes[tname].value = enable
        self._sync_plots()

      # Grouped checkboxes.
      groups = _group_terms(term_names)
      for group_label, members in groups.items():
        use_folder = len(groups) > 1
        ctx = (
          self._server.gui.add_folder(group_label, expand_by_default=False)
          if use_folder
          else contextlib.nullcontext()
        )

        with ctx:
          for tname in members:
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

  def _sync_plots(self) -> None:
    """Create or remove plots to match current selection."""
    for state in self._terms.values():
      if state.enabled and state.plot is None:
        self._create_plot(state)
      elif not state.enabled and state.plot is not None:
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
