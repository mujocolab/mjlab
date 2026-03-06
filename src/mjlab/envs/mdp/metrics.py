from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

def mean_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Record the mean action acceleration."""
  print(f"action is{env.action_manager.action}")
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  print(f"action acc is {action_acc}")
  print(f"mean action acc is {torch.mean(action_acc)}")
  return torch.mean(action_acc)