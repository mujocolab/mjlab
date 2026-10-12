"""Tests for HoldBuffer."""

import pytest
import torch
from conftest import get_test_device

from mjlab.utils.buffers import HoldBuffer


@pytest.fixture
def device():
  return get_test_device()


def _refresh_steps(buffer: HoldBuffer, num_steps: int, batch: int, device: str):
  """Feed the step index as data and return, per row, the steps that refreshed."""
  steps: list[list[int]] = [[] for _ in range(batch)]
  for k in range(num_steps):
    held = buffer.compute(torch.full((batch, 1), float(k), device=device))
    for row in range(batch):
      if held[row, 0].item() == k:
        steps[row].append(k)
  return steps


def test_hold_buffer_30hz_on_50hz(device):
  """Rate 0.6 refreshes with 2, 2, 1 step gaps; the phase shifts the pattern."""
  phase = torch.tensor([0.01, 0.5], dtype=torch.float64, device=device)
  buffer = HoldBuffer(rate=0.6, phase=phase)
  steps = _refresh_steps(buffer, 11, 2, device)
  assert steps[0] == [0, 2, 4, 5, 7, 9, 10]
  assert steps[1] == [0, 1, 3, 5, 6, 8, 10]


@pytest.mark.parametrize(
  "rate,expected", [(0.5, [0, 2, 4, 6]), (1.0, list(range(7))), (2.5, list(range(7)))]
)
def test_hold_buffer_rates(device, rate, expected):
  phase = torch.zeros(1, dtype=torch.float64, device=device)
  assert _refresh_steps(HoldBuffer(rate=rate, phase=phase), 7, 1, device)[0] == expected


def test_hold_buffer_reset_and_backfill(device):
  """A reset row refreshes on backfill and on its next step; others keep their
  schedule."""
  phase = torch.zeros(2, dtype=torch.float64, device=device)
  buffer = HoldBuffer(rate=0.25, phase=phase)
  buffer.compute(torch.zeros(2, 1, device=device))
  buffer.compute(torch.ones(2, 1, device=device))

  buffer.reset(torch.tensor([1], device=device))
  held = buffer.backfill(
    torch.full((2, 1), 2.0, device=device), torch.tensor([1], device=device)
  )
  assert held[:, 0].tolist() == [0.0, 2.0]

  held = buffer.compute(torch.full((2, 1), 3.0, device=device))
  assert held[:, 0].tolist() == [0.0, 3.0]
  # Backfill did not advance row 0, so it refreshes on its 5th step.
  held = buffer.compute(torch.full((2, 1), 4.0, device=device))
  assert held[:, 0].tolist() == [0.0, 3.0]
  held = buffer.compute(torch.full((2, 1), 5.0, device=device))
  assert held[:, 0].tolist() == [5.0, 3.0]


def test_hold_buffer_rejects_non_positive_rate(device):
  with pytest.raises(ValueError, match="rate must be > 0"):
    HoldBuffer(rate=0.0, phase=torch.zeros(1, device=device))
