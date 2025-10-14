"""Circular buffer for storing a history of batched tensor data."""

from __future__ import annotations

import torch


class CircularBuffer:
  """Circular buffer for storing a history of batched tensor data.

  Stores history in a circular buffer with shape (max_len, batch_size, ...).
  Returns history in oldest-to-newest order.
  """

  def __init__(self, max_len: int, batch_size: int, device: str) -> None:
    """Initialize the circular buffer.

    Args:
      max_len: The maximum length of the circular buffer.
      batch_size: The batch dimension of the data.
      device: The device used for processing.
    """
    if max_len < 1:
      raise ValueError(f"Buffer size must be >= 1, got {max_len}")

    self._max_len = max_len
    self._batch_size = batch_size
    self._device = device
    self._pointer: int = -1  # -1 means not initialized.
    self._buffer: torch.Tensor | None = None

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
  def buffer(self) -> torch.Tensor:
    """Get buffer in oldest-to-newest order. Shape: (batch_size, max_length, ...)."""
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")

    buf = torch.roll(self._buffer, shifts=self._max_len - self._pointer - 1, dims=0)
    return torch.transpose(buf, dim0=0, dim1=1)

  def reset(self, batch_ids: torch.Tensor | None = None) -> None:
    """Reset buffer for specified batch indices.

    Args:
      batch_ids: Tensor of batch indices to reset, or None to reset all.
    """
    if self._buffer is None:
      return

    if batch_ids is None:
      self._buffer.zero_()
    else:
      self._buffer[:, batch_ids] = 0.0

  def append(self, data: torch.Tensor) -> None:
    """Append data to the circular buffer.

    Args:
      data: Tensor of shape (batch_size, ...) to append.
    """
    if data.shape[0] != self._batch_size:
      raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

    data = data.to(self._device)

    if self._buffer is None:
      self._pointer = -1
      self._buffer = data.unsqueeze(0).repeat(
        self._max_len, 1, *([1] * (data.ndim - 1))
      )

    self._pointer = (self._pointer + 1) % self._max_len
    self._buffer[self._pointer] = data
