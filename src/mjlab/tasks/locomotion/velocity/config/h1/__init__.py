import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Flat-Unitree-H1-v0",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeH1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeH1FlatPPORunnerCfg",
  },
)
