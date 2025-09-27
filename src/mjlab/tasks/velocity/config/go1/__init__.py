import gymnasium as gym

# Import the config instances
from .flat_env_cfg import UNITREE_GO1_FLAT_ENV_CFG, UNITREE_GO1_FLAT_ENV_CFG_PLAY
from .rough_env_cfg import UNITREE_GO1_ROUGH_ENV_CFG, UNITREE_GO1_ROUGH_ENV_CFG_PLAY

gym.register(
  id="Mjlab-Velocity-Rough-Unitree-Go1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_GO1_ROUGH_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Rough-Unitree-Go1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_GO1_ROUGH_ENV_CFG_PLAY,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Unitree-Go1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_GO1_FLAT_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Unitree-Go1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": UNITREE_GO1_FLAT_ENV_CFG_PLAY,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)
