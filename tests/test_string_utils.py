"""Tests for mjlab.utils.string pattern helpers."""

import pytest

from mjlab.utils.string import filter_exp, resolve_expr

NAMES = ("joint1", "joint2", "joint10", "joint11")


def test_resolve_expr_requires_full_match():
  """A key matches whole names only, so "joint1" does not hit joint10/11."""
  assert resolve_expr({"joint1": 0.5, ".*": 0.0}, NAMES) == (0.5, 0.0, 0.0, 0.0)


def test_filter_exp_requires_full_match():
  assert filter_exp(["joint1"], NAMES) == ("joint1",)


def test_resolve_expr_raises_on_unmatched_key():
  """A key that matches no name is a typo or a prefix; fail loudly."""
  with pytest.raises(ValueError, match="joint_1"):
    resolve_expr({"joint_1": 0.5, ".*": 0.0}, NAMES)
