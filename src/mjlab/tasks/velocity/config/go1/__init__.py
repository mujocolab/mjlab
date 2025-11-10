import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Rough-Unitree-Go1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.env_cfgs:UNITREE_GO1_ROUGH_ENV_CFG",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Unitree-Go1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.env_cfgs:UNITREE_GO1_FLAT_ENV_CFG",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1PPORunnerCfg",
  },
)
