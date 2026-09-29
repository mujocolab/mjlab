"""Live per-node bar-chart snapshot for the Robot Health tab.

Adapts the HTML/CSS bar-rendering technique from
``src/mjlab/viewer/viser/reward_bar_panel.py`` (reused for its approach,
not imported): that panel colors bars by *sign* (reward terms are
positive/negative), which doesn't fit values that are always positive
(temperature, current, torque) where what matters is *magnitude relative
to a healthy range*. ``NodeBarPanel`` colors each bar on a green->red
gradient by how far it sits past a configurable ``(low, high)`` range,
and always shows every node as its own row -- there is no "top N" cutoff
hiding lower values, since the point is to see all nodes simultaneously.

This does not flag, threshold, or classify anything -- it's a visual aid
only, for a human watching the live view to notice where degradation
concentrates before any automated failure detection exists.
"""

from __future__ import annotations

import html

import numpy as np
import viser

_HEALTHY_COLOR = (0x4C, 0xAF, 0x50)  # green
_DANGER_COLOR = (0xF4, 0x43, 0x36)  # red


def _severity(value: float, low: float, high: float) -> float:
  """0.0 = within [low, high]; grows to 1.0 one range-width past a bound."""
  span = max(high - low, 1e-9)
  if value > high:
    return min(1.0, (value - high) / span)
  if value < low:
    return min(1.0, (low - value) / span)
  return 0.0


def _lerp_color(severity: float) -> str:
  t = max(0.0, min(1.0, severity))
  rgb = tuple(
    round(a + (b - a) * t) for a, b in zip(_HEALTHY_COLOR, _DANGER_COLOR, strict=True)
  )
  return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class NodeBarPanel:
  """HTML bar panel showing every node's current value, colored by danger."""

  def __init__(
    self,
    server: viser.ViserServer,
    node_names: list[str],
    low: float,
    high: float,
    unit: str = "",
  ) -> None:
    self._server = server
    self._node_names = node_names
    self._low = low
    self._high = high
    self._unit = unit
    self._values = [0.0] * len(node_names)
    self._html_handle = self._server.gui.add_html("")
    self._render_empty()

  def update(self, values: np.ndarray) -> None:
    """Replace displayed values and re-render. ``values`` shape: (num_nodes,)."""
    self._values = [float(v) for v in values]
    self._render()

  def clear(self) -> None:
    self._values = [0.0] * len(self._node_names)
    self._render_empty()

  def cleanup(self) -> None:
    self._html_handle.remove()

  def _render_empty(self) -> None:
    self._html_handle.content = (
      '<div style="padding:0.5em;color:#999;font-size:0.85em;">Waiting for data…</div>'
    )

  def _render(self) -> None:
    display_max = self._high * 1.25 if self._high > 0 else 1.0

    rows: list[str] = []
    for name, val in zip(self._node_names, self._values, strict=True):
      severity = _severity(val, self._low, self._high)
      color = _lerp_color(severity)
      pct = max(0.0, min(1.0, val / display_max)) * 100.0
      text_color = "#fff" if pct > 25 else "#ccc"
      val_str = f"{val:.3f}{self._unit}"
      safe_name = html.escape(name, quote=True)

      rows.append(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="min-width:110px;font-size:0.72em;text-align:right;'
        f"padding-right:6px;color:#ddd;white-space:nowrap;overflow:hidden;"
        f'text-overflow:ellipsis;" title="{safe_name}">{safe_name}</span>'
        f'<div style="flex:1;background:#333;border-radius:3px;height:16px;'
        f'position:relative;overflow:hidden;">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{color};'
        f'border-radius:3px;transition:width 0.15s,background 0.15s;"></div>'
        f'<span style="position:absolute;right:4px;top:0;line-height:16px;'
        f'font-size:0.7em;color:{text_color};">{val_str}</span>'
        f"</div></div>"
      )

    markup = (
      '<div style="padding:0.3em 0.5em;font-family:monospace;">'
      + "".join(rows)
      + "</div>"
    )
    self._html_handle.content = markup
