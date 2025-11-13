import gymnasium as gym

from .env_cfgs import (
  UNITREE_G1_FLAT_TRACKING_ENV_CFG,
  UNITREE_G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_G1_FLAT_TRACKING_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1FlatPPORunnerCfg",
  },
)


gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1FlatPPORunnerCfg",
  },
)
