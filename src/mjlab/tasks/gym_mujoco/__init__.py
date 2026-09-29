"""Native manager-based Gymnasium MuJoCo v5 tasks."""

from mjlab.tasks.gym_mujoco.config import TASK_CONFIGS
from mjlab.tasks.gym_mujoco.config.rl_cfg import gym_mujoco_ppo_cfg
from mjlab.tasks.registry import register_mjlab_task

for task_id, env_cfg in TASK_CONFIGS.items():
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=gym_mujoco_ppo_cfg(task_id),
  )
