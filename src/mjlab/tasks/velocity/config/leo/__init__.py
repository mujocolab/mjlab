from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  ccbr_leo_flat_env_cfg,
  ccbr_leo_flat_env_cfg_learned,
  ccbr_leo_rough_env_cfg,
)
from .rl_cfg import ccbr_leo_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-CCBR-Leo",
  env_cfg=ccbr_leo_rough_env_cfg(),
  play_env_cfg=ccbr_leo_rough_env_cfg(play=True),
  rl_cfg=ccbr_leo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-CCBR-Leo",
  env_cfg=ccbr_leo_flat_env_cfg(),
  play_env_cfg=ccbr_leo_flat_env_cfg(play=True),
  rl_cfg=ccbr_leo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-CCBR-Leo-ActuatorNet",
  env_cfg=ccbr_leo_flat_env_cfg_learned(),
  play_env_cfg=ccbr_leo_flat_env_cfg_learned(play=True),
  rl_cfg=ccbr_leo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
