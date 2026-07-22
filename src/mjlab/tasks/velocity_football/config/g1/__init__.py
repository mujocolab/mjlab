"""Unitree G1 velocity-football task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity_football.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_g1_flat_env_cfg
from .rl_cfg import (
  unitree_g1_ppo_runner_cfg,
  unitree_g1_velocity_pretrain_ppo_runner_cfg,
)
from .velocity_env_cfgs import unitree_g1_velocity_pretrain_flat_env_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Football-Pretrain-Flat-Unitree-G1",
  env_cfg=unitree_g1_velocity_pretrain_flat_env_cfg(),
  play_env_cfg=unitree_g1_velocity_pretrain_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_velocity_pretrain_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Football-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
