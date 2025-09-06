"""Reward plotting functionality for Viser viewer."""

from collections import deque

import numpy as np
import viser
import viser.uplot


class ViserRewardPlotter:
  """Handles reward plotting for the Viser viewer with individual plots per term."""

  def __init__(
    self,
    server: viser.ViserServer,
    history_length: int = 150,
    update_interval: int = 5,
    max_terms: int = 12,
  ):
    """Initialize the reward plotter.

    Args:
        server: The Viser server instance
        history_length: Number of points to keep in history
        update_interval: Update plots every N frames
        max_terms: Maximum number of reward terms to plot
    """
    self._server = server
    self._history_length = history_length
    self._update_interval = update_interval
    self._max_terms = max_terms

    # State
    self._term_names: list[str] = []
    self._histories: dict[str, deque[float]] = {}
    self._plot_handles: dict[str, viser.GuiUplotHandle] = {}

    # Pre-allocated x-axis array (reused for all plots)
    self._x_array = np.arange(-history_length + 1, 1, dtype=np.float64)

    self._update_counter = 0
    self._visible = True
    self._folder_handle = None

  def initialize(self, term_names: list[str]) -> None:
    """Initialize individual plots for each reward term.

    Args:
        term_names: List of reward term names to plot
    """
    # Clean up existing plots
    for handle in self._plot_handles.values():
      handle.remove()
    self._plot_handles.clear()
    self._histories.clear()

    self._term_names = term_names[: self._max_terms]

    if not self._term_names:
      return

    # Create individual plot for each reward term
    for name in self._term_names:
      # Initialize history deque for this term
      self._histories[name] = deque(maxlen=self._history_length)

      # Create initial empty data
      x_data = np.array([], dtype=np.float64)
      y_data = np.array([], dtype=np.float64)

      # Configure series for this single term
      series = [
        viser.uplot.Series(label="Steps"),  # X-axis
        viser.uplot.Series(
          label=name,
          stroke="#1f77b4",  # Blue for all plots
          width=2,
        ),
      ]

      # Create uPlot chart for this term with title
      plot_handle = self._server.gui.add_uplot(
        data=(x_data, y_data),
        series=tuple(series),
        scales={
          "x": viser.uplot.Scale(
            time=False, auto=False, range=(-self._history_length, 0)
          ),
          "y": viser.uplot.Scale(auto=True),
        },
        legend=viser.uplot.Legend(show=False),  # No legend needed for single series
        title=name,  # Add title to the plot
        aspect=2.0,  # Wider aspect ratio for individual plots
        visible=self._visible,
      )

      self._plot_handles[name] = plot_handle

  def update(self, reward_terms: list[tuple[str, np.ndarray]]) -> None:
    """Update the plots with new reward data.

    Args:
        reward_terms: List of (name, value_array) tuples
    """
    if not self._plot_handles or not self._term_names:
      return

    # Only update at specified interval
    self._update_counter += 1
    if self._update_counter % self._update_interval != 0:
      return

    # Update each term's plot individually
    for name, arr in reward_terms:
      if name not in self._histories or name not in self._plot_handles:
        continue

      value = float(arr[0])
      if np.isfinite(value):
        # Add to history deque (automatically pops oldest when full)
        self._histories[name].append(value)

        # Update this term's plot
        hist = self._histories[name]
        hist_len = len(hist)

        if hist_len > 0:
          # Use view of pre-allocated x-array
          x_data = self._x_array[-hist_len:]

          # Convert deque to numpy array efficiently
          # np.fromiter is efficient for converting iterables
          y_data = np.fromiter(hist, dtype=np.float64, count=hist_len)

          # Update plot data
          self._plot_handles[name].data = (x_data, y_data)

  def clear_histories(self) -> None:
    """Clear all reward histories."""
    for history in self._histories.values():
      history.clear()

    # Reset plot data to empty
    for handle in self._plot_handles.values():
      handle.data = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))

  def set_visible(self, visible: bool) -> None:
    """Set visibility of all plots.

    Args:
        visible: Whether plots should be visible
    """
    self._visible = visible
    for handle in self._plot_handles.values():
      handle.visible = visible

  def cleanup(self) -> None:
    """Clean up resources."""
    for handle in self._plot_handles.values():
      handle.remove()
    self._plot_handles.clear()
    self._histories.clear()
    self._term_names.clear()
