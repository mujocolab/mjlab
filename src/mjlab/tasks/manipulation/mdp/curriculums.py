from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  # verify that the stage is only either defined as step or iteration
  if any("iteration" in stage and "step" in stage for stage in weight_stages):
    raise ValueError("stage should only be defined as step or iteration, not both")
  for stage in weight_stages:
    if "step" in stage:
      steps = stage["step"]
    else:
      steps = stage["iteration"] * env.cfg.num_steps_per_env
    if env.common_step_counter > steps:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
