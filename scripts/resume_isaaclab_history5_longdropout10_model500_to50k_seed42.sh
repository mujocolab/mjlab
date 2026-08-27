#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
source_run="2026-08-18_17-32-05_IsaacLab_history5_mask_longdropout10_flat_seed42_from_walk16000_to50k_wandb"
source_checkpoint="model_500.pt"
run_name="IsaacLab_history5_mask_longdropout10_flat_seed42_resume500_to50k_wandb"
checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football/$source_run/$source_checkpoint"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing resume checkpoint: $checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

uv run train \
  Mjlab-Velocity-Football-IsaacLabAligned-History5-LongDropout10-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.max-iterations 49500 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
