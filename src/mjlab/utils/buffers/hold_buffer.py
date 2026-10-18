"""Sample-and-hold buffer for observations refreshed below the control rate."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class HoldBuffer:
  """Refresh rows at a fixed rate and hold the last value in between.

  A row refreshes on its first step after creation or reset, then whenever
  ``floor(k * rate + phase)`` increments, where ``k`` counts steps since reset.

  Args:
    rate: Refreshes per step. Rates >= 1 refresh every step.
    phase: Per-row phase in [0, 1), shape (batch_size,). Held by reference.
  """

  def __init__(self, rate: float, phase: torch.Tensor) -> None:
    if rate <= 0:
      raise ValueError(f"rate must be > 0, got {rate}")
    self.rate = rate
    self._phase = phase
    self._step = torch.zeros(phase.shape[0], dtype=torch.long, device=phase.device)
    self._held: torch.Tensor | None = None

  @property
  def is_initialized(self) -> bool:
    """Check if buffer has been initialized with at least one compute."""
    return self._held is not None

  def compute(self, data: torch.Tensor) -> torch.Tensor:
    """Advance one step and return the held data.

    Args:
      data: Fresh observation of shape (batch_size, ...).

    Returns:
      Held observation with shape (batch_size, ...).
    """
    k = self._step.double()
    due = (self._step == 0) | (
      torch.floor(k * self.rate + self._phase)
      > torch.floor((k - 1) * self.rate + self._phase)
    )
    self._step += 1
    if self._held is None:
      self._held = data.clone()
      return self._held
    self._held = torch.where(due.view(-1, *[1] * (data.ndim - 1)), data, self._held)
    return self._held

  def backfill(self, data: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
    """Refresh the given rows as their first step, without advancing other rows.

    Used after a partial reset. The refresh counts as the reset rows' first step,
    so their schedule matches a reset followed by compute().

    Args:
      data: Tensor of shape (batch_size, ...); only rows at batch_ids are read.
      batch_ids: Batch indices to refresh.

    Returns:
      Held observation with shape (batch_size, ...).
    """
    if self._held is None:
      raise RuntimeError("Buffer not initialized. Call compute() first.")
    self._step[batch_ids] = 1
    self._held = self._held.index_put((batch_ids,), data[batch_ids])
    return self._held

  def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    """Make the given rows refresh on their next step.

    Args:
      batch_ids: Batch indices to reset, or None to reset all.
    """
    self._step[slice(None) if batch_ids is None else batch_ids] = 0
