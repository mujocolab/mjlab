#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
walk_checkpoint="/home/ut/football_project/log_old/logs/rsl_rl/g1_velocity_football_pretrain/2026-07-23_18-17-07/model_16000.pt"
run_name="B1_A1R0_longdropout10_isaac_actor_dr_flat_seed42_from_walk16000_to50k_wandb"

if [[ ! -f "$walk_checkpoint" ]]; then
  echo "ERROR: missing walking checkpoint: $walk_checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

uv run train \
  Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-LegacyCurriculum-Flat-Unitree-G1 \
  --pretrained-checkpoint "$walk_checkpoint" \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 50000 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.upload-model False \
  --agent.run-name "$run_name"
