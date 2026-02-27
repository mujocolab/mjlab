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
  individual_plot: viser.GuiUplotHandle | None = None


class ViserTermPlotter:
  """Handles plotting for the Viser viewer with selective display and combined overlay."""

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

    # Pre-allocated x-axis array (reused for all plots)
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
    self._overlay_handle: viser.GuiUplotHandle | None = None

    # Dummy series for empty state (viser requires x + >=1 y).
    self._dummy_series = (
      viser.uplot.Series(label="Steps"),
      viser.uplot.Series(label="\u2014", stroke="#888", width=1),
    )
    self._empty = np.array([], dtype=np.float64)

    # Build all GUI elements while we're in the correct tab context.
    self._build_selector_gui(term_names)
    self._build_overlay_plot()
    # Folder for individual plots — created now (in the right tab) so that
    # lazily-added plots from callbacks are scoped correctly.
    self._plots_folder = self._server.gui.add_folder(
      self._label("Individual"), expand_by_default=True
    )

  def _label(self, text: str) -> str:
    """Namespace a GUI label to avoid collisions between plotters."""
    return f"{self._name}: {text}"

  def _build_selector_gui(self, term_names: list[str]) -> None:
    """Build grouped checkboxes for term selection."""
    with self._server.gui.add_folder(
      self._label("Select terms"), expand_by_default=True
    ):
      # Bulk actions.
      bulk = self._server.gui.add_button_group(
        self._label("bulk"), options=["All", "None"]
      )

      @bulk.on_click
      def _(event) -> None:
        enable = event.target.value == "All"
        for tname, state in self._terms.items():
          state.enabled = enable
          self._checkboxes[tname].value = enable
        self._sync_individual_plots()
        self._refresh_overlay()

      # Grouped checkboxes.
      groups = _group_terms(term_names)
      for group_label, members in groups.items():
        use_folder = len(groups) > 1
        ctx = (
          self._server.gui.add_folder(self._label(group_label), expand_by_default=False)
          if use_folder
          else contextlib.nullcontext()
        )

        with ctx:
          for tname in members:
            state = self._terms[tname]
            cb = self._server.gui.add_checkbox(
              self._label(tname),
              initial_value=state.enabled,
              hint=f"Color: {state.color}",
            )
            self._checkboxes[tname] = cb

            @cb.on_update
            def _(event, _tname=tname) -> None:
              self._terms[_tname].enabled = event.target.value
              self._sync_individual_plots()
              self._refresh_overlay()

  def _build_overlay_plot(self) -> None:
    """Create the combined overlay uPlot."""
    self._overlay_handle = self._server.gui.add_uplot(
      data=(self._empty, self._empty),
      series=self._dummy_series,
      scales={
        "x": viser.uplot.Scale(
          time=False, auto=False, range=(-self._history_length, 0)
        ),
        "y": viser.uplot.Scale(auto=True),
      },
      legend=viser.uplot.Legend(show=True),
      title=self._label("overlay"),
      aspect=2.5,
      visible=True,
    )

  def _sync_individual_plots(self) -> None:
    """Create or remove individual plots to match current selection."""
    for state in self._terms.values():
      if state.enabled and state.individual_plot is None:
        self._create_individual_plot(state)
      elif not state.enabled and state.individual_plot is not None:
        state.individual_plot.remove()
        state.individual_plot = None

  def _create_individual_plot(self, state: _TermState) -> None:
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
      state.individual_plot = self._server.gui.add_uplot(
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
        title=self._label(state.name),
        aspect=2.0,
        visible=True,
      )

  def _refresh_overlay(self) -> None:
    """Rebuild overlay plot series/data for currently-enabled terms."""
    if self._overlay_handle is None:
      return

    enabled = [s for s in self._terms.values() if s.enabled]
    if not enabled:
      self._overlay_handle.series = self._dummy_series
      self._overlay_handle.data = (self._empty, self._empty)
      return

    max_len = max(len(s.history) for s in enabled)
    if max_len == 0:
      return

    x = self._x_array[-max_len:]

    series = [viser.uplot.Series(label="Steps")]
    data: list[np.ndarray] = [x]

    for s in enabled:
      series.append(viser.uplot.Series(label=s.name, stroke=s.color, width=2))
      arr = np.full(max_len, np.nan, dtype=np.float64)
      if len(s.history) > 0:
        vals = np.fromiter(s.history, dtype=np.float64, count=len(s.history))
        arr[-len(vals) :] = vals
      data.append(arr)

    self._overlay_handle.series = tuple(series)
    self._overlay_handle.data = tuple(data)

  def update(self, terms: list[tuple[str, np.ndarray]]) -> None:
    """Push new data and refresh visible plots.

    Args:
        terms: ``[(name, value_array), ...]`` from the reward/metrics manager.
    """
    # Always accumulate history even for hidden terms so toggling on
    # immediately shows full history.
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

    # Update overlay.
    self._refresh_overlay()

    # Update individual plots.
    for state in self._terms.values():
      if not state.enabled or state.individual_plot is None:
        continue
      h = state.history
      hist_len = len(h)
      if hist_len > 0:
        x = self._x_array[-hist_len:]
        y = np.fromiter(h, dtype=np.float64, count=hist_len)
        state.individual_plot.data = (x, y)

  def clear_histories(self) -> None:
    """Clear all histories (e.g. on env reset or env switch)."""
    for state in self._terms.values():
      state.history.clear()
      if state.individual_plot is not None:
        state.individual_plot.data = (self._empty, self._empty)
    self._refresh_overlay()

  def cleanup(self) -> None:
    """Remove all GUI handles."""
    if self._overlay_handle is not None:
      self._overlay_handle.remove()
    for state in self._terms.values():
      if state.individual_plot is not None:
        state.individual_plot.remove()
    for cb in self._checkboxes.values():
      cb.remove()
    self._terms.clear()
    self._checkboxes.clear()
