"""Tests for CircularBuffer."""

import pytest
import torch

from mjlab.utils.buffers import CircularBuffer


def get_test_device():
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda:0"
  return "cpu"


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


def test_circular_buffer_basic_append(device):
  """Basic append and chronological retrieval (oldest -> newest)."""
  buffer = CircularBuffer(max_len=3, batch_size=2, device=device)

  buffer.append(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device))
  buffer.append(torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device))
  buffer.append(torch.tensor([[9.0, 10.0], [11.0, 12.0]], device=device))

  result = buffer.buffer
  assert result.shape == (2, 3, 2)
  # Oldest to newest.
  assert torch.allclose(
    result[0], torch.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]], device=device)
  )
  assert torch.allclose(
    result[1], torch.tensor([[3.0, 4.0], [7.0, 8.0], [11.0, 12.0]], device=device)
  )


def test_circular_buffer_overwrite(device):
  """Overwrites oldest once capacity reached."""
  buffer = CircularBuffer(max_len=2, batch_size=1, device=device)

  buffer.append(torch.tensor([[1.0]], device=device))
  buffer.append(torch.tensor([[2.0]], device=device))
  buffer.append(torch.tensor([[3.0]], device=device))  # Overwrites first.

  result = buffer.buffer
  assert result.shape == (1, 2, 1)
  assert torch.allclose(result[0], torch.tensor([[2.0], [3.0]], device=device))


def test_circular_buffer_reset_single_batch(device):
  """Reset clears values and counters for specified batch rows."""
  buffer = CircularBuffer(max_len=2, batch_size=3, device=device)

  buffer.append(torch.tensor([[1.0], [2.0], [3.0]], device=device))
  buffer.append(torch.tensor([[4.0], [5.0], [6.0]], device=device))

  # Reset only batch index 1.
  buffer.reset(batch_ids=torch.tensor([1], device=device))

  result = buffer.buffer
  # Oldest-to-newest for each batch row.
  assert result[0, 0, 0] == 1.0
  assert result[1, 0, 0] == 0.0  # Reset backfilled to zeros for that row.
  assert result[2, 0, 0] == 3.0

  # current_length reflects reset: rows 0 and 2 had 2 pushes, row 1 is 0.
  cl = buffer.current_length
  assert torch.equal(cl.cpu(), torch.tensor([2, 0, 2]))


def test_circular_buffer_first_append_fills(device):
  """First append back-fills whole history for each batch row."""
  buffer = CircularBuffer(max_len=3, batch_size=2, device=device)
  buffer.append(torch.tensor([[1.0], [2.0]], device=device))

  result = buffer.buffer
  assert torch.allclose(result[0], torch.tensor([[1.0], [1.0], [1.0]], device=device))
  assert torch.allclose(result[1], torch.tensor([[2.0], [2.0], [2.0]], device=device))

  # And current_length reflects valid frames so far.
  cl = buffer.current_length
  assert torch.equal(cl.cpu(), torch.tensor([1, 1]))


def test_current_length_counts_and_clamps(device):
  """current_length counts per-batch valid frames and clamps to max_len."""
  buffer = CircularBuffer(max_len=4, batch_size=3, device=device)

  # Two appends -> length 2 everywhere.
  for _ in range(2):
    buffer.append(torch.arange(3, dtype=torch.float32, device=device).unsqueeze(-1))

  assert torch.equal(buffer.current_length.cpu(), torch.tensor([2, 2, 2]))

  # Reset middle row -> it becomes 0.
  buffer.reset(batch_ids=[1])
  assert torch.equal(buffer.current_length.cpu(), torch.tensor([2, 0, 2]))

  # One more append -> rows [0,2] become 3; row 1 becomes 1.
  buffer.append(torch.arange(3, dtype=torch.float32, device=device).unsqueeze(-1))
  assert torch.equal(buffer.current_length.cpu(), torch.tensor([3, 1, 3]))

  # Fill beyond capacity -> clamp to max_len.
  for _ in range(5):
    buffer.append(torch.arange(3, dtype=torch.float32, device=device).unsqueeze(-1))
  assert torch.equal(buffer.current_length.cpu(), torch.tensor([4, 4, 4]))


def test_reset_all_none_path(device):
  """reset(None) zeros the entire buffer and counters without indexing."""
  buffer = CircularBuffer(max_len=3, batch_size=2, device=device)
  buffer.append(torch.tensor([[1.0], [2.0]], device=device))
  buffer.append(torch.tensor([[3.0], [4.0]], device=device))

  buffer.reset()  # None -> reset all.

  # Counters are zero.
  assert torch.equal(buffer.current_length.cpu(), torch.tensor([0, 0]))
  # Buffer zeros (safe to read even after reset because storage exists).
  result = buffer.buffer
  assert torch.count_nonzero(result) == 0


def test_getitem_lifo_and_clamp(device):
  """__getitem__ returns lagged frames per-batch (LIFO), clamping when needed."""
  buffer = CircularBuffer(max_len=3, batch_size=2, device=device)

  buffer.append(torch.tensor([[1.0], [10.0]], device=device))  # t0
  buffer.append(torch.tensor([[2.0], [20.0]], device=device))  # t1
  buffer.append(torch.tensor([[3.0], [30.0]], device=device))  # t2

  # Lag 0 for batch 0 (-> 3), lag 2 for batch 1 (-> oldest 10).
  out = buffer[torch.tensor([0, 2], device=device)]
  assert torch.allclose(out, torch.tensor([[3.0], [10.0]], device=device))

  # Clamp: huge lag for batch 0 -> oldest (1), lag 1 for batch 1 -> 20.
  out = buffer[torch.tensor([99, 1], device=device)]
  assert torch.allclose(out, torch.tensor([[1.0], [20.0]], device=device))


def test_backfill_after_per_batch_reset(device):
  """After resetting a row, the next append back-fills its entire history for that row."""
  buffer = CircularBuffer(max_len=3, batch_size=2, device=device)

  buffer.append(torch.tensor([[1.0], [10.0]], device=device))  # t0
  buffer.append(torch.tensor([[2.0], [20.0]], device=device))  # t1

  # Reset only batch row 1; row 0 remains with 2 valid frames.
  buffer.reset(batch_ids=[1])
  assert torch.equal(buffer.current_length.cpu(), torch.tensor([2, 0]))

  # Next append: row 0 gets new value; row 1 is "first push" -> back-filled.
  buffer.append(torch.tensor([[3.0], [99.0]], device=device))  # t2

  hist = buffer.buffer  # shape (2, 3, 1)
  # Row 0 keeps real chronology [1, 2, 3].
  assert torch.allclose(
    hist[0].squeeze(-1), torch.tensor([1.0, 2.0, 3.0], device=device)
  )
  # Row 1 is back-filled to all 99s.
  assert torch.allclose(
    hist[1].squeeze(-1), torch.tensor([99.0, 99.0, 99.0], device=device)
  )


def test_errors_and_types(device):
  """Error paths: wrong batch, pre-append access, and bad key size."""
  buffer = CircularBuffer(max_len=2, batch_size=2, device=device)

  # Wrong batch size on append.
  with pytest.raises(ValueError):
    buffer.append(torch.tensor([[1.0]], device=device))  # batch_size=1 wrong

  # buffer property before first append.
  with pytest.raises(RuntimeError):
    _ = CircularBuffer(max_len=1, batch_size=1, device=device).buffer

  # __getitem__ before any valid pushes.
  with pytest.raises(RuntimeError):
    _ = buffer[torch.tensor([0, 0], device=device)]

  # Now append once so storage exists and counters > 0.
  buffer.append(torch.tensor([[1.0], [2.0]], device=device))

  # __getitem__ with wrong key length.
  with pytest.raises(ValueError):
    _ = buffer[torch.tensor([0], device=device)]


def test_dtype_and_device_preserved(device):
  """Buffer preserves dtype and device."""
  buffer = CircularBuffer(max_len=2, batch_size=2, device=device)
  x = torch.tensor([[1.0], [2.0]], dtype=torch.float32, device=device)
  buffer.append(x)
  assert buffer.buffer.dtype == torch.float32
  assert buffer.buffer.device.type == torch.device(device).type
