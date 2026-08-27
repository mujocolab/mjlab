"""Depth-image asymmetric Actor--Critic football task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity_football.rl import VelocityOnPolicyRunner

from .env_cfg import (
  unitree_g1_depth_asymmetric_flat_env_cfg,
  unitree_g1_depth_auxiliary_flat_env_cfg,
  unitree_g1_depth_teacher_student_flat_env_cfg,
  unitree_g1_depth_temporal_calibrated_visual_dr_flat_env_cfg,
  unitree_g1_depth_temporal_deployment_robust_v2_flat_env_cfg,
  unitree_g1_depth_temporal_long_dropout10_camera_dr_flat_env_cfg,
  unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg,
  unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg,
  unitree_g1_depth_temporal_teacher_student_flat_env_cfg,
)
from .rl_cfg import (
  unitree_g1_depth_asymmetric_ppo_runner_cfg,
  unitree_g1_depth_auxiliary_ppo_runner_cfg,
  unitree_g1_depth_student_ppo_runner_cfg,
  unitree_g1_depth_teacher_distillation_runner_cfg,
  unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg,
  unitree_g1_depth_temporal_constrained_latent_runner_cfg,
  unitree_g1_depth_temporal_deployment_robust_v2_runner_cfg,
  unitree_g1_depth_temporal_teacher_distillation_runner_cfg,
)
from .runner import (
  DepthAuxVelocityOnPolicyRunner,
  DepthStudentPpoRunner,
  DepthTeacherDistillationRunner,
)

TASK_ID = "Mjlab-Velocity-Football-Depth-Asymmetric-Flat-Unitree-G1"
V1_TASK_ID = "Mjlab-Velocity-Football-Depth-Auxiliary-Flat-Unitree-G1"
DISTILLATION_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-Teacher-Distillation-Flat-Unitree-G1"
)
TEMPORAL_DISTILLATION_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-Distillation-Flat-Unitree-G1"
)
TEMPORAL_LONG_DROPOUT10_CAMERA_DR_DISTILLATION_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-LongDropout10-CameraDR-"
  "Distillation-Flat-Unitree-G1"
)
TEMPORAL_DEPLOYMENT_ROBUST_V2_DISTILLATION_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-DeploymentRobustV2-"
  "Distillation-Flat-Unitree-G1"
)
TEMPORAL_CALIBRATED_FROZEN_MLP_DISTILLATION_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-CalibratedVisualDR-"
  "FrozenMLP-Distillation-Flat-Unitree-G1"
)
TEMPORAL_MOUNT_RANGE_FROZEN_MLP_DISTILLATION_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeVisualDR-"
  "FrozenMLP-Distillation-Flat-Unitree-G1"
)
TEMPORAL_MOUNT_RANGE_STRONG_FROZEN_MLP_DISTILLATION_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-"
  "FrozenMLP-Distillation-Flat-Unitree-G1"
)
TEMPORAL_MOUNT_RANGE_STRONG_CONSTRAINED_DISTILLATION_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-"
  "ConstrainedMLP-Distillation-Flat-Unitree-G1"
)
STUDENT_PPO_TASK_ID = "Mjlab-Velocity-Football-Depth-Student-PPO-Flat-Unitree-G1"


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_depth_asymmetric_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_asymmetric_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_asymmetric_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=TEMPORAL_LONG_DROPOUT10_CAMERA_DR_DISTILLATION_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_long_dropout10_camera_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_long_dropout10_camera_dr_flat_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_depth_temporal_teacher_distillation_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=TEMPORAL_DEPLOYMENT_ROBUST_V2_DISTILLATION_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_deployment_robust_v2_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_deployment_robust_v2_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_temporal_deployment_robust_v2_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=TEMPORAL_CALIBRATED_FROZEN_MLP_DISTILLATION_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_calibrated_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_calibrated_visual_dr_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=TEMPORAL_MOUNT_RANGE_FROZEN_MLP_DISTILLATION_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=TEMPORAL_MOUNT_RANGE_STRONG_FROZEN_MLP_DISTILLATION_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=TEMPORAL_MOUNT_RANGE_STRONG_CONSTRAINED_DISTILLATION_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_depth_temporal_constrained_latent_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=V1_TASK_ID,
  env_cfg=unitree_g1_depth_auxiliary_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_auxiliary_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_auxiliary_ppo_runner_cfg(),
  runner_cls=DepthAuxVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=DISTILLATION_TASK_ID,
  env_cfg=unitree_g1_depth_teacher_student_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_teacher_student_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_teacher_distillation_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=TEMPORAL_DISTILLATION_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_teacher_student_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_teacher_student_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_temporal_teacher_distillation_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=STUDENT_PPO_TASK_ID,
  env_cfg=unitree_g1_depth_teacher_student_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_teacher_student_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_student_ppo_runner_cfg(),
  runner_cls=DepthStudentPpoRunner,
)
