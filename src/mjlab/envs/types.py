from __future__ import annotations

from typing import Dict

import torch

VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]

__all__ = (
  "VecEnvObs",
  "VecEnvStepReturn",
)
