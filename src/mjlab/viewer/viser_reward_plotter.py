# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    term_names: list[str],
    history_length: int = 150,
    max_terms: int = 12,
  ):
    """Initialize the reward plotter.

    Args:
        server: The Viser server instance
        term_names: List of reward term names to plot
        history_length: Number of points to keep in history
        max_terms: Maximum number of reward terms to plot
    """
    self._server = server
    self._history_length = history_length
    self._max_terms = max_terms

    # State
    self._term_names = term_names[: self._max_terms]
    self._histories: dict[str, deque[float]] = {}
    self._plot_handles: dict[str, viser.GuiUplotHandle] = {}

    # Pre-allocated x-axis array (reused for all plots)
    self._x_array = np.arange(-history_length + 1, 1, dtype=np.float64)
    self._folder_handle = None

    # Add checkbox to enable/disable reward plots
    self._enabled_checkbox = self._server.gui.add_checkbox(
      "Enable reward plots", initial_value=False
    )

    @self._enabled_checkbox.on_update
    def _(_) -> None:
      # Show/hide plots based on checkbox state
      for handle in self._plot_handles.values():
        handle.visible = self._enabled_checkbox.value

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
        visible=False,
      )

      self._plot_handles[name] = plot_handle

  def update(self, reward_terms: list[tuple[str, np.ndarray]]) -> None:
    """Update the plots with new reward data.

    Args:
        reward_terms: List of (name, value_array) tuples
    """
    # Early return if plots are disabled
    if not self._enabled_checkbox.value:
      return

    if not self._plot_handles or not self._term_names:
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
    for handle in self._plot_handles.values():
      handle.visible = visible

  def cleanup(self) -> None:
    """Clean up resources."""
    for handle in self._plot_handles.values():
      handle.remove()
    self._plot_handles.clear()
    self._histories.clear()
    self._term_names.clear()
