#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
distillation_checkpoint="${1:-}"
run_name="DepthStudentPPO_from_coordinate_teacher_distillation_seed42"

if [[ -z "$distillation_checkpoint" ]]; then
  echo "Usage: $0 /absolute/path/to/distillation/model_N.pt" >&2
  exit 2
fi
if [[ ! -f "$distillation_checkpoint" ]]; then
  echo "ERROR: missing distillation checkpoint: $distillation_checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

uv run train \
  Mjlab-Velocity-Football-Depth-Student-PPO-Flat-Unitree-G1 \
  --pretrained-checkpoint "$distillation_checkpoint" \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 30000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.upload-model False \
  --agent.run-name "$run_name"
