from dataclasses import dataclass

from mjlab.envs.manager_based_env_config import ManagerBasedEnvCfg


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg(ManagerBasedEnvCfg):
  episode_length_s: float
  rewards: object
  terminations: object
  commands: object | None = None
  curriculum: object | None = None
  is_finite_horizon: bool = False
