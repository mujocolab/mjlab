#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
source_run="2026-08-24_18-42-28_schemeA_ballcnn64_longdropout10_from_walk20000_seed42_50k_wandb"
source_checkpoint="model_41000.pt"
run_name="schemeA_ballcnn64_longdropout10_seed42_resume41000_to50k_pushAfterCmd95_save1000_wandb"
checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_klavier_ball_temporal/$source_run/$source_checkpoint"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing resume checkpoint: $checkpoint" >&2
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
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.max-iterations 9001 \
  --agent.save-interval 1000 \
  --agent.upload-model False \
  --agent.logger wandb \
  --agent.run-name "$run_name"
