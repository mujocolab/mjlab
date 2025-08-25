import gymnasium as gym

gym.register(
  id="Mjlab-Tracking-Flat-T1-v0",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:T1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:T1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-T1-Play-v0",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:T1FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:T1FlatPPORunnerCfg",
  },
)
