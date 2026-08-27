#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
source_run="2026-08-07_11-52-45_B1_A1R0_episode_loss5_no_ball_term_seed42_resume5k_to20k_wandb"
source_checkpoint="model_15000.pt"
run_name="B1_A1R0_episode_loss5_envelope30_w1_seed42_resume15k_to20k_wandb"
checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football/$source_run/$source_checkpoint"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing resume checkpoint: $checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

.venv/bin/train Mjlab-Velocity-Football-A1R0-Dropout5-Envelope30-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.max-iterations 5000 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.upload-model False \
  --agent.run-name "$run_name"
