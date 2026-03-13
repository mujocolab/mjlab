from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["RewardCurriculumStage", "reward_curriculum"]


class _RewardCurriculumStageOptional(TypedDict, total=False):
  weight: float
  params: dict[str, Any]


class RewardCurriculumStage(_RewardCurriculumStageOptional):
  step: int


def reward_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  stages: list[RewardCurriculumStage],
) -> dict[str, torch.Tensor]:
  """Update a reward term's weight and/or params based on training steps.

  Each stage specifies a ``step`` threshold and optionally a ``weight``
  and/or ``params`` dict.  When ``env.common_step_counter`` reaches a
  stage's ``step``, the corresponding values are applied.  Later stages
  take precedence when multiple thresholds are reached.

  Example::

    CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 12000, "weight": -0.1},
          {"step": 24000, "weight": -1.0, "params": {"max_vel": 1.0}},
        ],
      },
    )
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      if "weight" in stage:
        reward_term_cfg.weight = stage["weight"]
      if "params" in stage:
        unknown = stage["params"].keys() - reward_term_cfg.params.keys()
        if unknown:
          raise KeyError(
            f"reward_curriculum: stage at step {stage['step']} sets"
            f" unknown param(s) {unknown} on reward term"
            f" '{reward_name}'. Check for typos."
          )
        reward_term_cfg.params.update(stage["params"])
  result: dict[str, torch.Tensor] = {
    "weight": torch.tensor(reward_term_cfg.weight),
  }
  for k, v in reward_term_cfg.params.items():
    if isinstance(v, (int, float)):
      result[k] = torch.tensor(v)
    elif isinstance(v, torch.Tensor):
      result[k] = v
  return result
