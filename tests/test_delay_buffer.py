"""Tests for DelayBuffer."""

import pytest
import torch
from conftest import get_test_device

from mjlab.utils.buffers import DelayBuffer


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


def make_gen(seed: int, device: str) -> torch.Generator:
  """Create a seeded generator for reproducible tests."""
  gen = torch.Generator(device=device)
  gen.manual_seed(seed)
  return gen


##
# Basic behavior.
##


def test_delay_buffer_zero_lag(device):
  """Zero lag returns current observation."""
  buffer = DelayBuffer(min_lag=0, max_lag=0, batch_size=2, device=device)

  buffer.append(torch.tensor([[1.0], [2.0]], device=device))
  result = buffer.compute()
  assert torch.allclose(result, torch.tensor([[1.0], [2.0]], device=device))

  buffer.append(torch.tensor([[3.0], [4.0]], device=device))
  result = buffer.compute()
  assert torch.allclose(result, torch.tensor([[3.0], [4.0]], device=device))


def test_delay_buffer_constant_lag(device):
  """Constant lag returns observation from N steps ago."""
  buffer = DelayBuffer(min_lag=2, max_lag=2, batch_size=1, device=device)

  buffer.append(torch.tensor([[1.0]], device=device))
  result = buffer.compute()
  assert torch.allclose(result, torch.tensor([[1.0]], device=device))

  buffer.append(torch.tensor([[2.0]], device=device))
  result = buffer.compute()
  assert torch.allclose(result, torch.tensor([[1.0]], device=device))

  buffer.append(torch.tensor([[3.0]], device=device))
  result = buffer.compute()
  assert torch.allclose(result, torch.tensor([[1.0]], device=device))

  buffer.append(torch.tensor([[4.0]], device=device))
  result = buffer.compute()
  assert torch.allclose(result, torch.tensor([[2.0]], device=device))


def test_value_matches_lag_per_env(device):
  """Delayed values correctly match current lags."""
  B = 3
  buf = DelayBuffer(
    0, 2, batch_size=B, per_env=True, device=device, generator=make_gen(1234, device)
  )
  for t in range(6):
    buf.append(torch.full((B, 1), float(t), device=device))
    y = buf.compute()
    lags = buf.current_lags
    for e in range(B):
      eff = int(min(lags[e].item(), t))
      assert y[e].item() == float(t - eff)


##
# Lag sampling modes.
##


def test_delay_buffer_per_env_lags(device):
  """Per-env mode allows different lags per environment."""
  buffer = DelayBuffer(min_lag=0, max_lag=3, batch_size=4, per_env=True, device=device)

  for i in range(10):
    buffer.append(torch.full((4, 1), float(i), device=device))
    buffer.compute()

  lags = buffer.current_lags
  assert torch.all(lags >= 0)
  assert torch.all(lags <= 3)


def test_delay_buffer_shared_lags(device):
  """Shared mode uses same lag across all environments."""
  buffer = DelayBuffer(min_lag=0, max_lag=3, batch_size=4, per_env=False, device=device)

  for i in range(10):
    buffer.append(torch.full((4, 1), float(i), device=device))
    buffer.compute()

  lags = buffer.current_lags
  assert torch.all(lags == lags[0])


##
# Hold probability.
##


def test_hold_prob_always_hold(device):
  """hold_prob=1.0 keeps lag frozen."""
  buf = DelayBuffer(
    0,
    3,
    batch_size=1,
    device=device,
    update_period=1,
    hold_prob=1.0,
    generator=make_gen(7, device),
  )
  for t in range(2):
    buf.append(torch.tensor([[float(t)]], device=device))
    buf.compute()
  first = buf.current_lags.item()
  for t in range(2, 20):
    buf.append(torch.tensor([[float(t)]], device=device))
    buf.compute()
    assert buf.current_lags.item() == first


def test_hold_prob_never_hold(device):
  """hold_prob=0.0 allows lag to change."""
  buf = DelayBuffer(
    0,
    3,
    batch_size=1,
    device=device,
    update_period=1,
    hold_prob=0.0,
    generator=make_gen(42, device),
  )
  for t in range(2):
    buf.append(torch.tensor([[float(t)]], device=device))
    buf.compute()
  prev, changed = buf.current_lags.item(), False
  for t in range(2, 20):
    buf.append(torch.tensor([[float(t)]], device=device))
    buf.compute()
    cur = buf.current_lags.item()
    if cur != prev:
      changed = True
      break
    prev = cur
  assert changed


##
# Update period.
##


def test_delay_buffer_update_period(device):
  """Update period controls how often lags are resampled."""
  buffer = DelayBuffer(
    min_lag=0,
    max_lag=3,
    batch_size=1,
    update_period=3,
    per_env_phase=False,
    device=device,
  )

  lag_history = []
  for i in range(12):
    buffer.append(torch.tensor([[float(i)]], device=device))
    buffer.compute()
    lag_history.append(buffer.current_lags[0].item())

  assert len(lag_history) == 12


def test_update_period_changes_only_on_schedule(device):
  """Lags update only at scheduled intervals."""
  buf = DelayBuffer(
    0,
    10,  # Wider range makes value changes more likely
    batch_size=1,
    update_period=3,
    per_env_phase=False,
    device=device,
    generator=make_gen(123, device),
  )

  # Track when lags are updated by checking step_count timing
  lag_values = []
  for t in range(12):
    buf.append(torch.tensor([[float(t)]], device=device))
    buf.compute()
    lag_values.append(buf.current_lags.item())

  # Verify lag stays constant between update periods
  # Update happens at step 0, 3, 6, 9
  # So: [0-2] same, [3-5] same, [6-8] same, [9-11] same
  assert lag_values[0] == lag_values[1] == lag_values[2]
  assert lag_values[3] == lag_values[4] == lag_values[5]
  assert lag_values[6] == lag_values[7] == lag_values[8]
  assert lag_values[9] == lag_values[10] == lag_values[11]


##
# Reset behavior.
##


def test_delay_buffer_reset_all(device):
  """Reset clears all environments."""
  buffer = DelayBuffer(min_lag=1, max_lag=2, batch_size=2, device=device)

  for i in range(5):
    buffer.append(torch.full((2, 1), float(i), device=device))
    buffer.compute()

  buffer.reset()

  assert torch.all(buffer.current_lags == 0)
  assert torch.all(buffer._step_count == 0)


def test_delay_buffer_reset_partial(device):
  """Reset with batch_ids only resets specified environments."""
  buffer = DelayBuffer(min_lag=0, max_lag=3, batch_size=4, device=device)

  for _ in range(5):
    buffer.append(torch.arange(4, device=device).unsqueeze(1).float())
    buffer.compute()

  lags_before = buffer.current_lags.clone()

  buffer.reset(batch_ids=torch.tensor([0, 2], device=device))

  lags_after = buffer.current_lags

  assert lags_after[0] == 0
  assert lags_after[2] == 0
  assert lags_after[1] == lags_before[1]
  assert lags_after[3] == lags_before[3]


def test_partial_reset_then_backfill(device):
  """Partial reset returns zeros until next append backfills."""
  B = 3
  buf = DelayBuffer(1, 2, batch_size=B, device=device, generator=make_gen(9, device))
  for t in range(3):
    x = torch.arange(1, B + 1, device=device).float().unsqueeze(1) * 10 + t
    buf.append(x)
    buf.compute()
  buf.reset(batch_ids=torch.tensor([1], device=device))
  y = buf.compute()
  assert torch.allclose(y[1], torch.zeros_like(y[1]))
  x_new = torch.tensor([[111.0], [999.0], [333.0]], device=device)
  buf.append(x_new)
  y2 = buf.compute()
  assert torch.allclose(y2[1], torch.tensor([[999.0]], device=device))


##
# Error handling.
##


def test_delay_buffer_not_initialized(device):
  """Compute before append raises error."""
  buffer = DelayBuffer(min_lag=0, max_lag=3, batch_size=1, device=device)

  with pytest.raises(RuntimeError, match="Buffer not initialized"):
    buffer.compute()


def test_append_wrong_batch_raises(device):
  """Appending wrong batch size raises error."""
  buf = DelayBuffer(0, 1, batch_size=2, device=device)
  with pytest.raises(ValueError):
    buf.append(torch.zeros(3, 1, device=device))


def test_delay_buffer_validation(device):
  """Input validation catches invalid parameters."""
  with pytest.raises(ValueError, match="min_lag must be >= 0"):
    DelayBuffer(min_lag=-1, max_lag=3, batch_size=1, device=device)

  with pytest.raises(ValueError, match="max_lag.*must be >= min_lag"):
    DelayBuffer(min_lag=5, max_lag=3, batch_size=1, device=device)

  with pytest.raises(ValueError, match="hold_prob must be in"):
    DelayBuffer(min_lag=0, max_lag=3, batch_size=1, hold_prob=1.5, device=device)

  with pytest.raises(ValueError, match="update_period must be >= 0"):
    DelayBuffer(min_lag=0, max_lag=3, batch_size=1, update_period=-1, device=device)


# peek_append: read-only counterpart to append + compute.


def test_peek_append_matches_real_compute_constant_lag(device):
  """peek_append equals a real append+compute (constant lag), mutating nothing.

  With min_lag == max_lag, the per-step resampling in compute() is
  deterministic, so peek_append (which reuses the current lags and resamples
  nothing) must produce identical values to a real append+compute.
  """
  torch.manual_seed(0)
  batch_size = 3
  peeked = DelayBuffer(min_lag=2, max_lag=2, batch_size=batch_size, device=device)
  # Warm the buffer through several real steps so lags/state have settled.
  for i in range(5):
    peeked.append(torch.full((batch_size, 2), float(i), device=device))
    peeked.compute()

  new_frame = torch.randn(batch_size, 2, device=device)

  assert peeked._buffer._buffer is not None

  # Reference: clone the internal state and perform the real append+compute.
  reference = DelayBuffer(min_lag=2, max_lag=2, batch_size=batch_size, device=device)
  reference._buffer._buffer = peeked._buffer._buffer.clone()
  reference._buffer._pointer = peeked._buffer._pointer
  reference._buffer._num_pushes = peeked._buffer._num_pushes.clone()
  reference._current_lags = peeked._current_lags.clone()
  reference._step_count = peeked._step_count.clone()
  reference.append(new_frame)
  expected = reference.compute()

  # Snapshot mutable state to assert peek_append leaves it untouched.
  pointer = peeked._buffer._pointer
  pushes = peeked._buffer._num_pushes.clone()
  storage = peeked._buffer._buffer.clone()
  lags = peeked._current_lags.clone()
  step_count = peeked._step_count.clone()

  result = peeked.peek_append(new_frame)

  assert torch.allclose(result, expected)
  assert peeked._buffer._pointer == pointer
  assert torch.equal(peeked._buffer._num_pushes, pushes)
  assert torch.equal(peeked._buffer._buffer, storage)
  assert torch.equal(peeked._current_lags, lags)
  assert torch.equal(peeked._step_count, step_count)


def test_peek_append_clamps_to_available_history(device):
  """peek_append clamps the lag to available history early in an episode."""
  buffer = DelayBuffer(min_lag=3, max_lag=3, batch_size=1, device=device)
  buffer.append(torch.tensor([[1.0]], device=device))
  buffer.compute()  # current_lags becomes 3, but only 1 frame exists.

  # One real frame in history; peeking a second can return at most the oldest.
  result = buffer.peek_append(torch.tensor([[2.0]], device=device))
  assert torch.allclose(result, torch.tensor([[1.0]], device=device))
