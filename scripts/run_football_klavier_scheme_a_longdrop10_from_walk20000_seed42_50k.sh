#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
walk_checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_walk_klavier_replica/2026-08-24_11-38-37_unitree_g1_flat_copied_model_seed42_30k_wandb/model_20000.pt"

if [[ ! -f "$walk_checkpoint" ]]; then
  echo "ERROR: missing Walk checkpoint: $walk_checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export WARP_CACHE_PATH=/tmp/mjlab-warp
export MPLCONFIGDIR=/tmp/mjlab-mpl
export XDG_CACHE_HOME=/tmp/mjlab-xdg

exec env \
  -u VIRTUAL_ENV \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u http_proxy \
  -u https_proxy \
  -u ALL_PROXY \
  -u all_proxy \
  WANDB_MODE=online \
  WANDB_INIT_TIMEOUT=180 \
  "$repo_dir/.venv/bin/train" \
  Mjlab-Velocity-Football-KlavierReplica-BallTemporal-Flat-Unitree-G1 \
  --pretrained-checkpoint "$walk_checkpoint" \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 50001 \
  --agent.save-interval 200 \
  --agent.upload-model False \
  --agent.logger wandb \
  --agent.run-name schemeA_ballcnn64_longdropout10_from_walk20000_seed42_50k_wandb
