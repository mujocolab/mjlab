"""Active depth-football distillation task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
  unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg,
  unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg,
)
from .rl_cfg import (
  unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg,
  unitree_g1_depth_temporal_constrained_latent_runner_cfg,
)
from .runner import DepthTeacherDistillationRunner

DEPTH_BASELINE_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeVisualDR-"
  "FrozenMLP-Distillation-Flat-Unitree-G1"
)
DEPTH_CANDIDATE_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-"
  "ConstrainedMLP-Distillation-Flat-Unitree-G1"
)


register_mjlab_task(
  task_id=DEPTH_BASELINE_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=DEPTH_CANDIDATE_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_depth_temporal_constrained_latent_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)
