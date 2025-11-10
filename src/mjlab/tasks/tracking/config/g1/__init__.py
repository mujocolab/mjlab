from mjlab.tasks.registry import register

from .env_cfgs import (
  G1_FLAT_TRACKING_ENV_CFG,
  G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
)
from .rl_cfg import UNITREE_G1_TRACKING_PPO_RUNNER_CFG

register(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=G1_FLAT_TRACKING_ENV_CFG,
  rl_cfg=UNITREE_G1_TRACKING_PPO_RUNNER_CFG,
)

register(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
  rl_cfg=UNITREE_G1_TRACKING_PPO_RUNNER_CFG,
)
