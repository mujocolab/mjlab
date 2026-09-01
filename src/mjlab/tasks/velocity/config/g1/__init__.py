from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityDistillationRunner, VelocityOnPolicyRunner

from .distillation_cfg import (
  unitree_g1_distillation_env_cfg,
  unitree_g1_distillation_runner_cfg,
  unitree_g1_finetune_env_cfg,
  unitree_g1_finetune_runner_cfg,
)
from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Stage 1: distill the rough teacher (with height scan) into a blind
# recurrent student via DAgger. Requires --agent.teacher-checkpoints.
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1-Distill",
  env_cfg=unitree_g1_distillation_env_cfg(),
  play_env_cfg=unitree_g1_distillation_env_cfg(play=True),
  rl_cfg=unitree_g1_distillation_runner_cfg(),
  runner_cls=VelocityDistillationRunner,
)

# Stage 2: RL fine-tune the distilled student with asymmetric PPO.
# Requires --agent.init-checkpoint pointing at a distillation checkpoint.
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1-Distill-Finetune",
  env_cfg=unitree_g1_finetune_env_cfg(),
  play_env_cfg=unitree_g1_finetune_env_cfg(play=True),
  rl_cfg=unitree_g1_finetune_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
