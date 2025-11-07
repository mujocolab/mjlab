import gymnasium as gym

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-Demo",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_DEMO",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatNoStateEstimationEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatNoStateEstimationEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)
