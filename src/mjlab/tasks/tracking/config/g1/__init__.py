import gymnasium as gym

# Import the config instances
from .flat_env_cfg import G1_FLAT_ENV_CFG, G1_FLAT_ENV_CFG_PLAY

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": G1_FLAT_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": G1_FLAT_ENV_CFG_PLAY,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)
