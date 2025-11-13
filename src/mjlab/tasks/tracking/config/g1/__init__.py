from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  UNITREE_G1_FLAT_TRACKING_ENV_CFG,
  UNITREE_G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
)
from .rl_cfg import UNITREE_G1_TRACKING_PPO_RUNNER_CFG

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=UNITREE_G1_FLAT_TRACKING_ENV_CFG,
  rl_cfg=UNITREE_G1_TRACKING_PPO_RUNNER_CFG,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=UNITREE_G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
  rl_cfg=UNITREE_G1_TRACKING_PPO_RUNNER_CFG,
)
