from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def mean_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean absolute action acceleration.

  Lower values indicate smoother actions.

  Returns:
    Per-environment scalar. Shape: ``(B,)``.
  """
  # Discrete second derivative: a_t - 2 * a_{t-1} + a_{t-2}.  (B, N)
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.mean(torch.abs(action_acc), dim=-1)  # (B,)
