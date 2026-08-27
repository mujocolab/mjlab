#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
teacher_dir="$repo_dir/logs/rsl_rl/g1_velocity_football/2026-08-15_22-26-20_IsaacLab_history5_mask_flat_seed42_from_walk16000_to50k_wandb"
teacher_checkpoint="$teacher_dir/model_49999.pt"
run_name="DepthTeacherDistillation_from_IsaacLabHistory5_model49999_seed42"

if [[ ! -f "$teacher_checkpoint" ]]; then
  echo "ERROR: missing coordinate Teacher: $teacher_checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

uv run train \
  Mjlab-Velocity-Football-Depth-Teacher-Distillation-Flat-Unitree-G1 \
  --pretrained-checkpoint "$teacher_checkpoint" \
  --env.scene.num-envs 1024 \
  --agent.seed 42 \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.upload-model False \
  --agent.run-name "$run_name"
