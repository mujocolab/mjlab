========================================
G1 football retained depth-policy lineage
========================================

Scope
=====

The active project keeps one coordinate Teacher, one frozen-MLP DepthStudent
baseline, and one constrained DepthStudent candidate. Checkpoint paths and
SHA-256 values are recorded in ``BASELINES.md`` at the repository root.
Historical direct-depth, auxiliary-PPO, and deployment-randomization ablations
are no longer registered tasks.

Coordinate Teacher
==================

``Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-LegacyCurriculum-Flat-Unitree-G1``
is the Teacher used by the retained DepthStudent lineage. Its Actor combines
five frames of proprioceptive/control observations with a ten-frame football
feature history. The football input is privileged simulation geometry, not a
camera measurement.

Frozen-MLP baseline
===================

``Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeVisualDR-FrozenMLP-Distillation-Flat-Unitree-G1``
is the retained DepthStudent baseline. It trains the depth encoder with
Teacher-controlled rollouts while keeping the coordinate-policy MLP frozen.
Camera installation interpolation uses ``alpha=0..0.25`` with small mount
residuals.

Run it with::

  ./scripts/run_depth_student_mount_range_frozen_mlp_teacher_rollout_10k_seed42.sh

Constrained candidate
=====================

``Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-ConstrainedMLP-Distillation-Flat-Unitree-G1``
uses stronger installation randomization, constrained latent adaptation, and
opens only the final MLP layer. Student-controlled rollout probability ramps to
30 percent.

Resume the retained candidate lineage with::

  ./scripts/resume_depth_student_mount_range_strong_constrained_model4000_to10k_seed42.sh

Evaluation boundary
===================

Compare both DepthStudents against the same coordinate Teacher, manifest,
command sequence, initial-ball states, and seeds. Report command tracking, ball
control, falls, ball-loss termination, action acceleration, and inference
latency. Teacher sim2sim may use true football position to establish the control
upper bound; camera and depth performance must be reported separately.
