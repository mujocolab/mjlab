import gymnasium as gym

gym.register(
  id="Tracking-Flat-G1-v0",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
  },
)
