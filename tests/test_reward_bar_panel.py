"""Tests for the Viser reward bar panel term cap."""

from unittest.mock import Mock

import pytest

from mjlab.viewer.viser.reward_bar_panel import RewardBarPanel


def _term_names(n: int) -> list[str]:
  return [f"term_{i}" for i in range(n)]


def test_warns_and_truncates_over_cap():
  """More terms than the cap warns and keeps only the first ``max_terms``."""
  server = Mock()
  with pytest.warns(UserWarning, match="exceed max_terms"):
    panel = RewardBarPanel(server, _term_names(21), update_dt=1 / 30, max_terms=20)
  assert panel._term_names == _term_names(21)[:20]


def test_no_warning_at_or_below_cap(recwarn):
  """At or below the cap, all terms are kept and no truncation warning fires."""
  server = Mock()
  panel = RewardBarPanel(server, _term_names(20), update_dt=1 / 30, max_terms=20)
  assert panel._term_names == _term_names(20)
  assert not [w for w in recwarn if "exceed max_terms" in str(w.message)]


def test_raised_cap_shows_all_terms():
  """Raising ``max_terms`` (the fix for #1079) surfaces the overflow terms."""
  server = Mock()
  panel = RewardBarPanel(server, _term_names(24), update_dt=1 / 30, max_terms=32)
  assert panel._term_names == _term_names(24)
