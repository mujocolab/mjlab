from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RewardWeightStage(TypedDict):
  step: int
  weight: float


class RewardParamStage(TypedDict):
  step: int
  params: dict[str, object]


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


def _deep_update(target: dict, source: dict) -> None:
  """Recursively merge ``source`` into ``target`` in-place.

  Dict-valued keys are merged rather than replaced, so callers can update a
  subset of a nested mapping without losing other entries.  All other value
  types are overwritten directly.
  """
  for key, value in source.items():
    if isinstance(value, dict) and isinstance(target.get(key), dict):
      _deep_update(target[key], value)
    else:
      target[key] = value


def reward_params(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param_stages: list[RewardParamStage],
) -> dict[str, torch.Tensor]:
  """Update a reward term's params based on training step stages.

  Each stage specifies a ``step`` threshold and a ``params`` dict with keys
  matching the reward function's keyword arguments.  When
  ``env.common_step_counter`` exceeds a stage's ``step``, the corresponding
  params are applied.  Later stages in the list take precedence when multiple
  thresholds are exceeded.

  When a param value is itself a dict (e.g. ``std_walking`` in a posture
  reward that maps joint-name patterns to std values), the stage value is
  **deep-merged** into the existing dict so that only the specified keys are
  updated and the rest are preserved.  Pass the full dict in the stage to
  replace it entirely.

  Example — scalar param::

    CurriculumTermCfg(
      func=reward_params,
      params={
        "reward_name": "track_linear_velocity",
        "param_stages": [
          {"step": 0,    "params": {"std": 0.5}},
          {"step": 1000, "params": {"std": 0.3}},
        ],
      },
    )

  Example — dict-valued param::

    CurriculumTermCfg(
      func=reward_params,
      params={
        "reward_name": "base_height",
        "param_stages": [
          {"step": 0,    "params": {"joint": {"leg_right_knee_joint": 0.5}}},
          {"step": 1000, "params": {"joint": {"leg_right_knee_joint": 0.3}}},
        ],
      },
    )
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in param_stages:
    if env.common_step_counter > stage["step"]:
      _deep_update(reward_term_cfg.params, stage["params"])
  return {
    k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
    for k, v in reward_term_cfg.params.items()
    if isinstance(v, (int, float, torch.Tensor))
  }
