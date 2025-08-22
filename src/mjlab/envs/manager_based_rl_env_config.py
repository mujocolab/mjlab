from dataclasses import dataclass
from typing import Any

from mjlab.envs.manager_based_env_config import ManagerBasedEnvCfg


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg(ManagerBasedEnvCfg):
  episode_length_s: float
  rewards: Any
  terminations: Any
  commands: Any | None = None
  curriculum: Any | None = None
  is_finite_horizon: bool = False
