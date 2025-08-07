from dataclasses import dataclass, MISSING

from mjlab.envs.manager_based_env_config import ManagerBasedEnvCfg


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg(ManagerBasedEnvCfg):
  is_finite_horizon: bool = False
  episode_length_s: float = MISSING
  rewards: object = MISSING
  terminations: object = MISSING
  commands: object | None = None
  curriculum: object | None = None
