from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

def modify_reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  term_name: str,
  step: int,
  weight: float,
) -> torch.Tensor:
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(term_name)
  assert reward_term_cfg is not None
  if env.common_step_counter > step:
    reward_term_cfg.weight = weight
  return torch.tensor(reward_term_cfg.weight)
