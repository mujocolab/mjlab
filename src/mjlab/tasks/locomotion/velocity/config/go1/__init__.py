import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Unitree-Go1-v0",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo1EnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo1FlatPPORunnerCfg",
  },
)
