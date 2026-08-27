#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
walk_checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_pretrain/2026-07-23_18-17-07/model_16000.pt"
run_name="B1_A1R0_visible_only_coordinate_teacher_flat_seed42_from_walk16000_to50k_wandb"

if [[ ! -f "$walk_checkpoint" ]]; then
  echo "ERROR: missing walking checkpoint: $walk_checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR=/tmp/uv-cache
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run train \
  Mjlab-Velocity-Football-A1R0-VisibleOnly-Envelope30-LegacyCurriculum-Flat-Unitree-G1 \
  --pretrained-checkpoint "$walk_checkpoint" \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 50000 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
