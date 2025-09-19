import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Unitree-Go1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo1EnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Unitree-Go1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo1EnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1FlatPPORunnerCfg",
  },
)
