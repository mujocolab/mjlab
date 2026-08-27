#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
source_run="2026-08-21_16-21-47_DepthStudent_MountRangeVisualDR_frozenMLP_teacherRollout_alpha025_noDelay_noLongDrop_pixel5_from_B1Teacher49999_seed42_10k_wandb"
source_checkpoint="model_4000.pt"
checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_depth_temporal_distillation/$source_run/$source_checkpoint"
run_name="DepthStudent_MountRangeStrongVisualDR_unfrozenMLP_mixedRollout_alpha035_seed42_resume4000_to10k_wandb"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing Student checkpoint: $checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR=/tmp/uv-cache
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run train \
  Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-FrozenMLP-Distillation-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.student.cnn-cfg.freeze-coordinate-actor False \
  --agent.algorithm.rollout-policy mixed \
  --agent.algorithm.student-rollout-warmup-updates 0 \
  --agent.algorithm.student-rollout-ramp-updates 2000 \
  --agent.algorithm.student-rollout-final-probability 1.0 \
  --agent.max-iterations 6000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
