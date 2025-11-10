from mjlab.envs.manager_based_rl_env import (
  ManagerBasedEnvCfg,
  ManagerBasedRlEnv,
  ManagerBasedRlEnvCfg,
)
from mjlab.envs.types import VecEnvObs, VecEnvStepReturn

# Backwards compatibility: ManagerBasedEnv is now just ManagerBasedRlEnv
ManagerBasedEnv = ManagerBasedRlEnv

__all__ = (
  "ManagerBasedRlEnvCfg",
  "ManagerBasedRlEnv",
  "ManagerBasedEnvCfg",
  "ManagerBasedEnv",
  "VecEnvStepReturn",
  "VecEnvObs",
)
