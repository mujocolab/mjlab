"""Circular buffer for storing a history of batched tensor data.

Understanding Dimensions
========================

Internal storage shape: (max_len, batch_size, ...)
                         ↑         ↑
                         time      environments
                         axis      axis

External view (buffer property): (batch_size, max_len, ...)
                                  ↑           ↑
                                  environments time
                                  axis        axis

Why different?
  - Internal: time-first for clean API (self._buffer[pointer] = data)
  - External: batch-first for typical use (iterate over environments)

Backfill Behavior
=================

When you first append to a new or reset batch row, that first value is copied
to ALL history slots for that row:

  buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")
  buffer.append(torch.tensor([[5.0], [10.0]]))

  # buffer.buffer contains (shape: 2, 3, 1):
  # Batch 0: [5.0, 5.0, 5.0]   <- all slots filled with first value
  # Batch 1: [10.0, 10.0, 10.0] <- all slots filled with first value

Why backfill?
  You always have valid data. If training expects 3 frames of history, you
  don't want garbage/zeros for the first 2 timesteps.

Per-Batch Reset
===============

Reset affects specific batch rows. The circular pointer advances globally, but
reset rows get "first-append" treatment on their next write:

  buffer = CircularBuffer(max_len=3, batch_size=3, device="cpu")

  buffer.append(torch.tensor([[1.0], [10.0], [100.0]]))   # t0
  buffer.append(torch.tensor([[2.0], [20.0], [200.0]]))   # t1
  buffer.append(torch.tensor([[3.0], [30.0], [300.0]]))   # t2

  # buffer.buffer:
  # Batch 0: [1.0, 2.0, 3.0]
  # Batch 1: [10.0, 20.0, 30.0]
  # Batch 2: [100.0, 200.0, 300.0]

  buffer.reset(batch_ids=[1])  # Reset only batch 1

  # buffer.buffer after reset:
  # Batch 0: [1.0, 2.0, 3.0]       <- unchanged
  # Batch 1: [0.0, 0.0, 0.0]       <- zeroed
  # Batch 2: [100.0, 200.0, 300.0] <- unchanged

  # current_length: [3, 0, 3]  <- batch 1 has 0 valid frames

  buffer.append(torch.tensor([[4.0], [99.0], [400.0]]))  # t3

  # buffer.buffer after append:
  # Batch 0: [2.0, 3.0, 4.0]        <- oldest overwritten (normal)
  # Batch 1: [99.0, 99.0, 99.0]     <- BACKFILLED with 99.0
  # Batch 2: [200.0, 300.0, 400.0]  <- oldest overwritten (normal)

Key insight:
  Reset only affects specific batch rows. The pointer keeps advancing for
  everyone, but reset rows get backfilled on their next append.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class CircularBuffer:
  """Fixed-length circular buffer for batched tensor history.

  Stores history with shape (max_len, batch_size, ...) internally.
  The `buffer` property returns chronologically ordered data (oldest to newest)
  with shape (batch_size, max_len, ...).

  Storage and Retrieval
  ---------------------
  Internal storage (circular):
    ┌─────┬─────┬─────┐
    │  2  │  3  │  1  │  <- pointer at index 1 (newest = 3)
    └─────┴─────┴─────┘
     idx:0  idx:1  idx:2

  buffer property returns (chronological, oldest→newest):
    ┌─────┬─────┬─────┐
    │  1  │  2  │  3  │
    └─────┴─────┴─────┘

  LIFO Retrieval via __getitem__
  -------------------------------
  Given buffer with [1, 2, 3] (oldest to newest):
    buffer[lag=0] -> 3  (most recent)
    buffer[lag=1] -> 2  (one step back)
    buffer[lag=2] -> 1  (oldest)

  Per-Batch Reset
  ---------------
  After reset(batch_ids=[1]):
    Batch 0: [1, 2, 3]  (unchanged)
    Batch 1: [0, 0, 0]  (zeroed, current_length=0)
    Batch 2: [1, 2, 3]  (unchanged)

  Next append backfills reset rows with first value.

  Args:
    max_len: Maximum number of historical frames to retain.
    batch_size: Size of the batch dimension.
    device: Torch device for storage.
  """

  def __init__(self, max_len: int, batch_size: int, device: str) -> None:
    if max_len < 1:
      raise ValueError(f"Buffer size must be >= 1, got {max_len}")

    self._max_len = max_len
    self._batch_size = batch_size
    self._device = device
    self._pointer: int = -1
    self._buffer: torch.Tensor | None = None
    self._all_indices = torch.arange(batch_size, device=device)
    self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._max_len_tensor = torch.full(
      (batch_size,), max_len, dtype=torch.int, device=device
    )

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @property
  def device(self) -> str:
    return self._device

  @property
  def max_length(self) -> int:
    return self._max_len

  @property
  def current_length(self) -> torch.Tensor:
    """Per-batch count of valid frames. Shape: (batch_size,)."""
    return torch.minimum(self._num_pushes, self._max_len_tensor)

  @property
  def buffer(self) -> torch.Tensor:
    """History in chronological order (oldest to newest).

    Returns:
      Tensor of shape (batch_size, max_len, ...) where index 0 is oldest
      and index -1 is newest.
    """
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")

    buf = torch.roll(self._buffer, shifts=self._max_len - self._pointer - 1, dims=0)
    return torch.transpose(buf, dim0=0, dim1=1)

  def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    """Zero out values and counters for specified batch rows.

    Args:
      batch_ids: Batch indices to reset, or None to reset all.
    """
    ids: Sequence[int] | torch.Tensor | slice = (
      slice(None) if batch_ids is None else batch_ids
    )
    self._num_pushes[ids] = 0
    if self._buffer is not None:
      self._buffer[:, ids] = 0.0

  def append(self, data: torch.Tensor) -> None:
    """Append a new frame for all batch elements.

    Args:
      data: Tensor of shape (batch_size, ...).
    """
    if data.shape[0] != self._batch_size:
      raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

    data = data.to(self._device)

    if self._buffer is None:
      self._pointer = -1
      self._buffer = torch.empty(
        (self._max_len, *data.shape), dtype=data.dtype, device=self._device
      )

    self._pointer = (self._pointer + 1) % self._max_len
    self._buffer[self._pointer] = data

    # Backfill entire history with first frame for newly initialized batches.
    is_first_push = self._num_pushes == 0
    if torch.any(is_first_push):
      self._buffer[:, is_first_push] = data[is_first_push]

    self._num_pushes += 1

  def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
    """Retrieve lagged frames per batch (LIFO).

    Args:
      key: Per-batch lags to retrieve. Shape (batch_size,).

    Returns:
      Retrieved frames with shape (batch_size, ...).
    """
    if key.numel() != self._batch_size:
      raise ValueError(f"Expected {self._batch_size} lags, got {key.numel()}")
    if self._buffer is None or torch.any(self._num_pushes == 0):
      raise RuntimeError("Buffer is empty. Call append() first.")

    valid_keys = torch.minimum(key, self._num_pushes - 1)
    idx = torch.remainder(self._pointer - valid_keys, self._max_len)
    return self._buffer[idx, self._all_indices]
