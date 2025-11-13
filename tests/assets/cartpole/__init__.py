"""CartPole test fixtures - robot and task definitions."""

import gymnasium as gym

from .cartpole_env_cfg import CARTPOLE_ENV_CFG

# Register the test environment
gym.register(
  id="Mjlab-Cartpole",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": CARTPOLE_ENV_CFG,
    "rl_cfg_entry_point": "mjlab.rl.config:RslRlOnPolicyRunnerCfg",
  },
)

