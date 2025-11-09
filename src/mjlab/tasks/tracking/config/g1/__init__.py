import gymnasium as gym

from .flat_env_cfg import (
  G1_FLAT_TRACKING_ENV_CFG as G1_FLAT_TRACKING_ENV_CFG,
)
from .flat_env_cfg import (
  G1_FLAT_TRACKING_ENV_CFG_DEMO as G1_FLAT_TRACKING_ENV_CFG_DEMO,
)
from .flat_env_cfg import (
  G1_FLAT_TRACKING_ENV_CFG_PLAY as G1_FLAT_TRACKING_ENV_CFG_PLAY,
)
from .flat_env_cfg import (
  G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG as G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
)
from .flat_env_cfg import (
  G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG_PLAY as G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG_PLAY,
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": G1_FLAT_TRACKING_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": G1_FLAT_TRACKING_ENV_CFG_PLAY,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-Demo",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": G1_FLAT_TRACKING_ENV_CFG_DEMO,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG_PLAY,
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1FlatPPORunnerCfg",
  },
)
