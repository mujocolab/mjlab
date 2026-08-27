#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
walk_checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_pretrain/2026-07-23_18-17-07/model_16000.pt"

cd "$repo_dir"

if [[ ! -f "$walk_checkpoint" ]]; then
  echo "ERROR: missing Walk checkpoint: $walk_checkpoint" >&2
  exit 1
fi

exec env \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u http_proxy \
  -u https_proxy \
  -u ALL_PROXY \
  -u all_proxy \
  WANDB_MODE=online \
  WANDB_INIT_TIMEOUT=180 \
  "$repo_dir/.venv/bin/train" \
  Mjlab-Velocity-Football-A1R0-History30-Flat-Unitree-G1 \
  --pretrained-checkpoint "$walk_checkpoint" \
  --env.scene.num-envs 4096 \
  --agent.upload-model False \
  --agent.seed 42 \
  --agent.max-iterations 20000 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.run-name B1_A1R0_history30_ballcritic_noise0p20_seed42_from_walk16000_football20k_wandb
