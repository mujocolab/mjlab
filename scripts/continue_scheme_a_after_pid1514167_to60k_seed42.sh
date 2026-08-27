#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
experiment="g1_velocity_football_klavier_ball_temporal"
source_run="2026-08-24_18-42-28_schemeA_ballcnn64_longdropout10_from_walk20000_seed42_50k_wandb"
source_dir="$repo_dir/logs/rsl_rl/$experiment/$source_run"
source_pid=1514167
target_iteration=60000
run_name="schemeA_ballcnn64_longdropout10_seed42_continue50k_to60k_push95_save1000_wandb"

while kill -0 "$source_pid" 2>/dev/null; do
  sleep 30
done

latest_checkpoint="$({
  find "$source_dir" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n'
} | sort -V | tail -1)"

if [[ ! "$latest_checkpoint" =~ ^model_([0-9]+)\.pt$ ]]; then
  echo "ERROR: unable to resolve final checkpoint below $source_dir" >&2
  exit 1
fi

source_iteration="${BASH_REMATCH[1]}"
if (( source_iteration >= target_iteration )); then
  echo "Target already reached: $latest_checkpoint"
  exit 0
fi

# RSL-RL starts its loop at the loaded iteration and saves the final loop index,
# so +1 is required to produce model_60000.pt exactly.
additional_iterations=$((target_iteration - source_iteration + 1))

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
  --agent.load-checkpoint "$latest_checkpoint" \
  --agent.max-iterations "$additional_iterations" \
  --agent.save-interval 1000 \
  --agent.upload-model False \
  --agent.logger wandb \
  --agent.run-name "$run_name"
