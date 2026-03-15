"""Real-time reward bar panel for the Viser viewer.

Renders horizontal bars for each reward term, sized by a running mean
over ~1 second. Positive terms are green, negative terms are red.
"""

from __future__ import annotations

import html
from collections import deque

import numpy as np
import viser


class RewardBarPanel:
  """HTML bar panel showing running-mean reward terms for quick comparison."""

  def __init__(
    self,
    server: viser.ViserServer,
    term_names: list[str],
    update_dt: float,
    max_terms: int = 20,
  ) -> None:
    """Initialize the reward bar panel.

    Args:
      server: The Viser server instance.
      term_names: List of reward term names.
      update_dt: Time in seconds between consecutive ``update()`` calls
        (i.e. the viewer's frame time). Used to size the averaging
        window to ~1 second.
      max_terms: Maximum number of terms to display.
    """
    self._server = server
    self._term_names = term_names[:max_terms]

    # ~1 second averaging window, but at least 1 step.
    self._window_steps = max(1, round(1.0 / update_dt))

    # Per-term circular buffer for running mean.
    self._histories: dict[str, deque[float]] = {
      name: deque(maxlen=self._window_steps) for name in self._term_names
    }

    self._html_handle = self._server.gui.add_html("")
    self._render_empty()

  # ── public API ──────────────────────────────────────────────────

  def update(self, terms: list[tuple[str, np.ndarray]]) -> None:
    """Push new values and re-render the bars.

    Args:
      terms: List of ``(term_name, value_array)`` tuples.
    """
    for name, arr in terms:
      if name not in self._histories:
        continue
      val = float(arr[0])
      if np.isfinite(val):
        self._histories[name].append(val)

    self._render()

  def clear_histories(self) -> None:
    """Clear all running-mean buffers and reset display."""
    for h in self._histories.values():
      h.clear()
    self._render_empty()

  def cleanup(self) -> None:
    """Remove the HTML element from the GUI."""
    self._html_handle.remove()

  # ── internal rendering ─────────────────────────────────────────

  def _render_empty(self) -> None:
    self._html_handle.content = (
      '<div style="padding:0.5em;color:#999;font-size:0.85em;">Waiting for data…</div>'
    )

  def _render(self) -> None:
    means: dict[str, float] = {}
    for name in self._term_names:
      buf = self._histories[name]
      if buf:
        means[name] = sum(buf) / len(buf)
      else:
        means[name] = 0.0

    max_abs = max((abs(v) for v in means.values()), default=1e-8)
    if max_abs < 1e-12:
      max_abs = 1e-12

    rows: list[str] = []
    for name in self._term_names:
      val = means[name]
      pct = abs(val) / max_abs * 100.0
      color = "#4caf50" if val >= 0 else "#f44336"  # green / red
      text_color = "#fff" if pct > 25 else "#ccc"

      # Value label — short scientific if tiny, else 4-decimal.
      if abs(val) < 1e-6 and val != 0:
        val_str = f"{val:.2e}"
      else:
        val_str = f"{val:.4f}"

      safe_name = html.escape(name, quote=True)
      rows.append(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        # Label
        f'<span style="min-width:120px;font-size:0.78em;text-align:right;'
        f"padding-right:6px;color:#ddd;white-space:nowrap;overflow:hidden;"
        f'text-overflow:ellipsis;" title="{safe_name}">{safe_name}</span>'
        # Bar container
        f'<div style="flex:1;background:#333;border-radius:3px;height:18px;'
        f'position:relative;overflow:hidden;">'
        # Bar fill
        f'<div style="width:{pct:.1f}%;height:100%;background:{color};'
        f'border-radius:3px;transition:width 0.15s;"></div>'
        # Numeric value overlay
        f'<span style="position:absolute;right:4px;top:0;line-height:18px;'
        f'font-size:0.72em;color:{text_color};">{val_str}</span>'
        f"</div></div>"
      )

    markup = (
      '<div style="padding:0.3em 0.5em;font-family:monospace;">'
      + "".join(rows)
      + "</div>"
    )
    self._html_handle.content = markup
