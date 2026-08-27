"""Active Unitree G1 velocity-football task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity_football.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg,
)
from .rl_cfg import unitree_g1_factorial_ppo_runner_cfg, unitree_g1_ppo_runner_cfg

BASE_TASK_ID = "Mjlab-Velocity-Football-Flat-Unitree-G1"
TEACHER_BASELINE_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-"
  "LegacyCurriculum-Flat-Unitree-G1"
)


register_mjlab_task(
  task_id=BASE_TASK_ID,
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=TEACHER_BASELINE_TASK_ID,
  env_cfg=unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg(),
  play_env_cfg=(
    unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg(play=True)
  ),
  rl_cfg=unitree_g1_factorial_ppo_runner_cfg(use_b1_history=True),
  runner_cls=VelocityOnPolicyRunner,
)
