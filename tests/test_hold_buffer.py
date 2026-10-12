"""Tests for HoldBuffer."""

import pytest
import torch
from conftest import get_test_device

from mjlab.utils.buffers import HoldBuffer


@pytest.fixture
def device():
  """Test device fixture."""
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
  """Integer periods refresh evenly and rates >= 1 refresh every step."""
  phase = torch.zeros(1, dtype=torch.float64, device=device)
  assert _refresh_steps(HoldBuffer(rate=rate, phase=phase), 7, 1, device)[0] == expected


def test_hold_buffer_reset_and_backfill(device):
  """A backfill refreshes only the reset row and counts as its first step; the
  other rows keep their schedule."""
  phase = torch.zeros(2, dtype=torch.float64, device=device)
  buffer = HoldBuffer(rate=0.25, phase=phase)
  buffer.compute(torch.zeros(2, 1, device=device))
  buffer.compute(torch.ones(2, 1, device=device))

  buffer.reset(torch.tensor([1], device=device))
  held = buffer.backfill(
    torch.full((2, 1), 2.0, device=device), torch.tensor([1], device=device)
  )
  assert held[:, 0].tolist() == [0.0, 2.0]

  for value in (3.0, 4.0):
    held = buffer.compute(torch.full((2, 1), value, device=device))
    assert held[:, 0].tolist() == [0.0, 2.0]
  # Row 0 refreshes on its 5th step, row 1 on its 4th step after the backfill.
  held = buffer.compute(torch.full((2, 1), 5.0, device=device))
  assert held[:, 0].tolist() == [5.0, 2.0]
  held = buffer.compute(torch.full((2, 1), 6.0, device=device))
  assert held[:, 0].tolist() == [5.0, 6.0]


def test_hold_buffer_backfill_matches_compute(device):
  """Reset then backfill (explicit reset) follows the same schedule as reset then
  compute (auto-reset)."""
  phase = torch.linspace(0.0, 0.95, 20, dtype=torch.float64, device=device)
  ids = torch.arange(20, device=device)
  auto = HoldBuffer(rate=0.3, phase=phase)
  explicit = HoldBuffer(rate=0.3, phase=phase)
  for buffer in (auto, explicit):
    buffer.compute(torch.zeros(20, 1, device=device))
    buffer.reset()

  auto.compute(torch.ones(20, 1, device=device))
  explicit.backfill(torch.ones(20, 1, device=device), ids)
  for k in range(2, 15):
    data = torch.full((20, 1), float(k), device=device)
    assert torch.equal(auto.compute(data), explicit.compute(data))
