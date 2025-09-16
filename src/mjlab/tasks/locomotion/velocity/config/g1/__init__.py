import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Unitree-G1-v0",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1EnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Unitree-G1-Play-v0",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1EnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1FlatPPORunnerCfg",
  },
)
