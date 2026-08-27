==========================================
G1 football direct-depth experiment design
==========================================

Initial task
============

``Mjlab-Velocity-Football-Depth-Asymmetric-Flat-Unitree-G1`` is a controlled
observation ablation of ``Mjlab-Velocity-Football-Flat-Unitree-G1``. Rewards,
commands, actions, events, curricula, and terminations are unchanged. The Actor
receives its existing 490-value proprioceptive history and one normalized
``80 x 60`` depth image. The Critic receives its clean proprioceptive state, the
same depth image, and the true yaw-frame football position.

The privileged football position is a separate current-frame two-value group;
it does not inherit the five-frame history used by the Critic proprioception.

The first version deliberately has no depth corruption, image history,
auxiliary perception loss, or pretrained-policy transfer. This keeps failures
attributable to the direct-depth observation and CNN rather than several new
mechanisms at once.

Start the initial run with::

  ./scripts/run_depth_asymmetric_v0_seed42.sh

The script starts with 512 environments because camera rendering is materially
more expensive than vector observations. Adjust the environment count only
after measuring GPU memory and steps per second on the target machine.

Temporal auxiliary task
=======================

``Mjlab-Velocity-Football-Depth-Auxiliary-Flat-Unitree-G1`` is the V1 task.
It keeps rewards and physics domain randomization unchanged while making four
targeted changes:

* the Actor is initialized from the 490-input walking checkpoint;
* five ``40 x 30`` depth frames, downsampled from the calibrated ``80 x 60``
  camera, are stacked as CNN channels;
* the encoder preserves spatial keypoints, feature magnitude, and direct depth
  statistics, then predicts normalized ball XY, planar distance, and visibility;
* training-only segmentation supplies visibility labels, and true ball state
  supplies geometry labels. Neither target is an Actor or exported-policy input.

The Critic remains asymmetric and receives the same depth history plus the
current true yaw-frame ball position. Start the 4,096-environment run with::

  ./scripts/run_depth_auxiliary_v1_from_walk_10k_seed42.sh

Coordinate-Teacher distillation
===============================

The distillation task freezes the completed 525-input History5 coordinate
Actor. A depth front-end predicts the final 35 football-history entries from
five ``40 x 30`` depth frames, concatenates them with the unchanged 490-input
proprioceptive history, and passes the reconstructed observation through an
identical copy of the Teacher MLP. Training uses visible-coordinate Smooth L1,
visibility BCE, and deterministic action Huber losses. Rollouts are controlled
by the Teacher during perception pretraining.

The default script uses
``2026-08-15_22-26-20_IsaacLab_history5_mask_flat_seed42_from_walk16000_to50k_wandb/model_49999.pt``::

  ./scripts/run_depth_teacher_distillation_seed42.sh

After selecting a distillation checkpoint, start low-learning-rate PPO
fine-tuning with::

  ./scripts/run_depth_student_ppo_from_distillation_seed42.sh \
    /absolute/path/to/distillation/model_N.pt

Experiment sequence
===================

Run experiments in order and retain seed 42 as the common seed. Add at least
seeds 43 and 44 before selecting a deployment candidate.

1. **V0 pipeline check.** Train the initial task for 1,000 iterations with a
   small environment count, verify image orientation and ball visibility, then
   benchmark camera-rendering throughput.
2. **Resolution ablation.** Compare ``40 x 30``, ``64 x 48``, and ``80 x 60``
   while keeping the temporal window and rollout budget fixed.
3. **Temporal-input ablation.** Compare one, three, and five frames stacked
   oldest-to-newest as channels. This tests whether explicit visual motion
   improves ball-velocity control.
4. **V3 observation corruption.** Add depth holes, distance-dependent noise,
   one-to-three-frame latency, frame freezing, and small camera-pose errors.
   Keep these settings separate from the existing physics domain randomization.
5. **V4 auxiliary perception.** Add training-only heads for yaw-frame ball
   position and visibility. Use Smooth L1 loss for visible ball position and
   binary cross entropy for visibility; do not add them to environment rewards.
6. **V5 teacher transfer.** Distill mean actions from the strongest coordinate-
   based football policy, decay the distillation loss, and finish with PPO-only
   fine-tuning.

Evaluation
==========

Report command tracking, ball velocity tracking, ball-control-zone success,
falls, ball-loss terminations, action acceleration, and inference latency.
Evaluate V3 and later under both clean simulated depth and held-out corrupted
depth. Before hardware deployment, validate the V1 exported Actor contract
``obs=(1, 490)``, ``depth=(1, 5, 30, 40)``, and ``actions=(1, 29)``.
