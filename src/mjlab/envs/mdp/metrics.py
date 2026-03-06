from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def mean_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Record the mean action acceleration."""
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.mean(action_acc, dim=-1)
