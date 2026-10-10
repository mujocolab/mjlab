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


@pytest.mark.parametrize("per_env", [True, False])
def test_hold_decision_matches_lag_sharing(device, per_env):
  """The hold decision is drawn per env, or once for the batch in shared mode."""
  B, steps = 256, 40
  buf = DelayBuffer(
    min_lag=0,
    max_lag=50,
    batch_size=B,
    per_env=per_env,
    hold_prob=0.5,
    device=device,
    generator=make_gen(0, device),
  )
  obs = torch.zeros(B, 1, device=device)
  buf.append(obs)
  buf.compute()

  num_held_steps = 0
  for _ in range(steps):
    prev = buf.current_lags.clone()
    buf.append(obs)
    buf.compute()
    num_held = int((buf.current_lags == prev).sum())
    if per_env:
      # About half of the envs hold on any given step.
      assert B // 4 < num_held < 3 * B // 4
    else:
      assert torch.all(buf.current_lags == buf.current_lags[0])
      num_held_steps += num_held == B
  if not per_env:
    # The whole batch holds on about half of the steps.
    assert steps // 8 < num_held_steps < 7 * steps // 8


@pytest.mark.parametrize("update_period", [0, 5])
def test_shared_lag_survives_partial_resets(device, update_period):
  """Partial resets do not pull envs off the shared lag or its schedule."""
  B = 8
  buf = DelayBuffer(
    min_lag=1,
    max_lag=5,
    batch_size=B,
    per_env=False,
    hold_prob=0.5,
    update_period=update_period,
    device=device,
    generator=make_gen(0, device),
  )
  obs = torch.zeros(B, 1, device=device)
  for t in range(60):
    if t % 3 == 0:
      buf.reset([t % B, (t + 3) % B])
    buf.append(obs)
    buf.compute()
    lags = buf.current_lags
    assert torch.all(lags == lags[0]) and lags[0] >= 1


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


@pytest.mark.parametrize(
  "kwargs",
  [
    {"update_period": 10, "per_env_phase": True},
    {"hold_prob": 0.5},
  ],
  ids=["staggered_update_period", "hold_prob"],
)
def test_lags_stay_within_range_from_first_step(device, kwargs):
  """Lags never fall below min_lag, including right after creation and reset."""
  B = 16
  buf = DelayBuffer(
    min_lag=2,
    max_lag=2,
    batch_size=B,
    device=device,
    generator=make_gen(0, device),
    **kwargs,
  )

  def run(start: int) -> None:
    for t in range(start, start + 4):
      buf.append(torch.full((B, 1), float(t), device=device))
      y = buf.compute()
      assert torch.all(buf.current_lags == 2), buf.current_lags.tolist()
      expected = float(max(t - 2, start))
      assert torch.all(y == expected), y.squeeze(-1).tolist()

  run(0)
  buf.reset()
  run(100)


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


def test_backfill_and_peek_leave_others_untouched(device):
  """backfill serves the reset row's new frame; peek advances no schedules."""
  B = 3
  buf = DelayBuffer(1, 2, batch_size=B, device=device, generator=make_gen(9, device))
  for t in range(4):
    x = torch.arange(1, B + 1, device=device).float().unsqueeze(1) * 10 + t
    buf.append(x)
    buf.compute()

  out_before = buf.peek()
  lags_before = buf.current_lags.clone()
  steps_before = buf._step_count.clone()

  reset_ids = torch.tensor([1], device=device)
  buf.reset(batch_ids=reset_ids)
  buf.backfill(torch.tensor([[-1.0], [999.0], [-1.0]], device=device), reset_ids)
  out = buf.peek()

  # Reset row serves its backfilled frame (lag zeroed, one valid frame).
  assert torch.allclose(out[1], torch.tensor([999.0], device=device))
  # Other rows: same delayed frame, lags and schedule untouched.
  keep = torch.tensor([0, 2], device=device)
  assert torch.allclose(out[keep], out_before[keep])
  assert torch.equal(buf.current_lags[keep], lags_before[keep])
  assert torch.equal(buf._step_count[keep], steps_before[keep])
