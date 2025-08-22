from dataclasses import dataclass
from typing import Generic

from mjlab.envs.manager_based_env_config import ManagerBasedEnvCfg
from mjlab.envs.types import (
  T_observations,
  T_actions,
  T_events,
  T_rewards,
  T_terminations,
  T_commands,
  T_curriculum,
)


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg(
  ManagerBasedEnvCfg[T_observations, T_actions, T_events],
  Generic[
    T_observations,
    T_actions,
    T_events,
    T_rewards,
    T_terminations,
    T_commands,
    T_curriculum,
  ],
):
  episode_length_s: float
  rewards: T_rewards
  terminations: T_terminations
  commands: T_commands | None = None
  curriculum: T_curriculum | None = None
  is_finite_horizon: bool = False
