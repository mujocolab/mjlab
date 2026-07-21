from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    booster_t1_velocity_env_cfg,
    booster_t1_velocity_play_env_cfg,
)
from .rl_cfg import booster_t1_velocity_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Booster-T1",
    env_cfg=booster_t1_velocity_env_cfg(),
    play_env_cfg=booster_t1_velocity_play_env_cfg(),
    rl_cfg=booster_t1_velocity_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
