import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Flat-Unitree-Go1-v0",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo1FlatEnvCfg",
    # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg",
  },
)
