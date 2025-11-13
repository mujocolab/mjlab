from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import UNITREE_GO1_FLAT_ENV_CFG, UNITREE_GO1_ROUGH_ENV_CFG
from .rl_cfg import UNITREE_GO1_PPO_RUNNER_CFG

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go1",
  env_cfg=UNITREE_GO1_ROUGH_ENV_CFG,
  rl_cfg=UNITREE_GO1_PPO_RUNNER_CFG,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1",
  env_cfg=UNITREE_GO1_FLAT_ENV_CFG,
  rl_cfg=UNITREE_GO1_PPO_RUNNER_CFG,
)
