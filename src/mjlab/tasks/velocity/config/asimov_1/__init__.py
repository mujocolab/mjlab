from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    asimov_1_flat_env_cfg,
    asimov_1_rough_env_cfg,
)
from .rl_cfg import asimov_1_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Asimov-1",
    env_cfg=asimov_1_rough_env_cfg(),
    play_env_cfg=asimov_1_rough_env_cfg(play=True),
    rl_cfg=asimov_1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Asimov-1",
    env_cfg=asimov_1_flat_env_cfg(),
    play_env_cfg=asimov_1_flat_env_cfg(play=True),
    rl_cfg=asimov_1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
