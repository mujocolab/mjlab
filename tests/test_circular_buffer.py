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
  """Test basic append and buffer retrieval."""
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
  """Test that buffer correctly overwrites oldest data."""
  buffer = CircularBuffer(max_len=2, batch_size=1, device=device)

  buffer.append(torch.tensor([[1.0]], device=device))
  buffer.append(torch.tensor([[2.0]], device=device))
  buffer.append(torch.tensor([[3.0]], device=device))  # Overwrites first.

  result = buffer.buffer
  assert result.shape == (1, 2, 1)
  assert torch.allclose(result[0], torch.tensor([[2.0], [3.0]], device=device))


def test_circular_buffer_reset(device):
  """Test that reset clears buffer for specified batches."""
  buffer = CircularBuffer(max_len=2, batch_size=3, device=device)

  buffer.append(torch.tensor([[1.0], [2.0], [3.0]], device=device))
  buffer.append(torch.tensor([[4.0], [5.0], [6.0]], device=device))

  buffer.reset(batch_ids=torch.tensor([1], device=device))

  result = buffer.buffer
  assert result[0, 0, 0] == 1.0
  assert result[1, 0, 0] == 0.0  # Reset.
  assert result[2, 0, 0] == 3.0


def test_circular_buffer_first_append_fills(device):
  """Test that first append initializes entire buffer."""
  buffer = CircularBuffer(max_len=3, batch_size=2, device=device)
  buffer.append(torch.tensor([[1.0], [2.0]], device=device))

  result = buffer.buffer
  # All history slots filled with first value.
  assert torch.allclose(result[0], torch.tensor([[1.0], [1.0], [1.0]], device=device))
  assert torch.allclose(result[1], torch.tensor([[2.0], [2.0], [2.0]], device=device))
