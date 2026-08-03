"""Unitree G1 velocity-football task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity_football.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  FactorialBallRewardVariant,
  RewardAblationVariant,
  unitree_g1_factorial_flat_env_cfg,
  unitree_g1_flat_env_cfg,
  unitree_g1_reward_ablation_flat_env_cfg,
  unitree_g1_temporal_flat_env_cfg,
  unitree_g1_temporal_history_flat_env_cfg,
  unitree_g1_temporal_stop_reward_flat_env_cfg,
  unitree_g1_visual_mask_flat_env_cfg,
)
from .rl_cfg import (
  unitree_g1_factorial_ppo_runner_cfg,
  unitree_g1_ppo_runner_cfg,
  unitree_g1_temporal_ppo_runner_cfg,
  unitree_g1_temporal_velocity_pretrain_ppo_runner_cfg,
  unitree_g1_velocity_pretrain_ppo_runner_cfg,
  unitree_g1_visual_mask_ppo_runner_cfg,
)
from .velocity_env_cfgs import (
  unitree_g1_current_velocity_pretrain_flat_env_cfg,
  unitree_g1_temporal_velocity_pretrain_flat_env_cfg,
  unitree_g1_velocity_pretrain_flat_env_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Football-Current-Pretrain-Flat-Unitree-G1",
  env_cfg=unitree_g1_current_velocity_pretrain_flat_env_cfg(),
  play_env_cfg=unitree_g1_current_velocity_pretrain_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_velocity_pretrain_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Football-Pretrain-Flat-Unitree-G1",
  env_cfg=unitree_g1_velocity_pretrain_flat_env_cfg(),
  play_env_cfg=unitree_g1_velocity_pretrain_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_velocity_pretrain_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Football-Temporal-Pretrain-Flat-Unitree-G1",
  env_cfg=unitree_g1_temporal_velocity_pretrain_flat_env_cfg(),
  play_env_cfg=unitree_g1_temporal_velocity_pretrain_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_temporal_velocity_pretrain_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

reward_ablation_tasks: tuple[tuple[str, RewardAblationVariant], ...] = (
  ("Mjlab-Velocity-Football-R0-IsaacLab-Robust-Flat-Unitree-G1", "r0_isaaclab"),
  ("Mjlab-Velocity-Football-R1-E1-Robust-Flat-Unitree-G1", "r1_e1"),
  (
    "Mjlab-Velocity-Football-R2-No-Relative-Velocity-Flat-Unitree-G1",
    "r2_no_relative_velocity",
  ),
  (
    "Mjlab-Velocity-Football-R3-No-Relative-Position-Flat-Unitree-G1",
    "r3_no_relative_position",
  ),
)

for task_id, variant in reward_ablation_tasks:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_g1_reward_ablation_flat_env_cfg(variant),
    play_env_cfg=unitree_g1_reward_ablation_flat_env_cfg(variant, play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )

register_mjlab_task(
  task_id="Mjlab-Velocity-Football-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Football-Temporal-Flat-Unitree-G1",
  env_cfg=unitree_g1_temporal_flat_env_cfg(),
  play_env_cfg=unitree_g1_temporal_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_temporal_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Football-Temporal-StopReward-Flat-Unitree-G1",
  env_cfg=unitree_g1_temporal_stop_reward_flat_env_cfg(),
  play_env_cfg=unitree_g1_temporal_stop_reward_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_temporal_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

for _history_length in (5, 20):
  register_mjlab_task(
    task_id=(
      f"Mjlab-Velocity-Football-Temporal-History{_history_length}-Flat-Unitree-G1"
    ),
    env_cfg=unitree_g1_temporal_history_flat_env_cfg(_history_length),
    play_env_cfg=unitree_g1_temporal_history_flat_env_cfg(
      _history_length,
      play=True,
    ),
    rl_cfg=unitree_g1_temporal_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )

register_mjlab_task(
  task_id="Mjlab-Velocity-Football-VisualMask-Flat-Unitree-G1",
  env_cfg=unitree_g1_visual_mask_flat_env_cfg(),
  play_env_cfg=unitree_g1_visual_mask_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_visual_mask_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

factorial_tasks: tuple[tuple[str, bool, FactorialBallRewardVariant], ...] = (
  (
    "Mjlab-Velocity-Football-A0R0-Flat-Unitree-G1",
    False,
    "r0_isaaclab_ball",
  ),
  (
    "Mjlab-Velocity-Football-A0R1-Flat-Unitree-G1",
    False,
    "r1_ball_center",
  ),
  (
    "Mjlab-Velocity-Football-A1R0-Flat-Unitree-G1",
    True,
    "r0_isaaclab_ball",
  ),
  (
    "Mjlab-Velocity-Football-A1R1-Flat-Unitree-G1",
    True,
    "r1_ball_center",
  ),
)

for task_id, use_b1_history, reward_variant in factorial_tasks:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_g1_factorial_flat_env_cfg(
      use_b1_history=use_b1_history,
      reward_variant=reward_variant,
    ),
    play_env_cfg=unitree_g1_factorial_flat_env_cfg(
      use_b1_history=use_b1_history,
      reward_variant=reward_variant,
      play=True,
    ),
    rl_cfg=unitree_g1_factorial_ppo_runner_cfg(
      use_b1_history=use_b1_history,
    ),
    runner_cls=VelocityOnPolicyRunner,
  )
