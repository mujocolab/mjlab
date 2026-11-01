"""Tests for the cartpole reward."""

import torch

from mjlab.tasks.cartpole.cartpole_env_cfg import _quadratic_tolerance


def test_quadratic_tolerance_matches_dm_control_small_control():
  """dm_control's small_control uses value_at_margin=0: 1 at 0, 0 at the margin."""
  x = torch.tensor([0.0, 0.5, 0.9, 1.0, 1.04])
  expected = torch.tensor([1.0, 0.75, 0.19, 0.0, 0.0])
  assert torch.allclose(_quadratic_tolerance(x, margin=1.0), expected, atol=1e-6)
