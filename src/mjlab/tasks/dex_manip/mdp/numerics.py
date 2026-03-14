from __future__ import annotations

import math

import torch


def sanitize_to_range(
  value: torch.Tensor | float,
  lower: float,
  upper: float,
  *,
  nan_default: float,
) -> torch.Tensor:
  tensor = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
  tensor = torch.nan_to_num(tensor, nan=nan_default, posinf=upper, neginf=lower)
  return torch.clamp(tensor, min=lower, max=upper)


def finite_or_default(value: float, default: float) -> float:
  return value if math.isfinite(value) else default


def safe_mean_finite(values: torch.Tensor) -> torch.Tensor | None:
  finite = values[torch.isfinite(values)]
  if finite.numel() == 0:
    return None
  return finite.mean()
