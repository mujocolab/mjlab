#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
current_pid="193360"
source_run="2026-08-06_17-05-32_B1_A1R0_dropout10_seed42_from_walk16000_football20k_wandb"
source_checkpoint="model_19999.pt"
run_name="B1_A1R0_dropout10_seed42_resume20k_to50k_wandb"

cd "$repo_dir"
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

echo "Waiting for PID $current_pid to finish the 0-20k stage."
while kill -0 "$current_pid" 2>/dev/null; do
  sleep 30
done

checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football/$source_run/$source_checkpoint"
if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: 0-20k stage ended without $checkpoint" >&2
  exit 1
fi

echo "Resuming dropout10 training from $checkpoint to model_49999.pt."
.venv/bin/train Mjlab-Velocity-Football-A1R0-Dropout10-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.max-iterations 30001 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.upload-model False \
  --agent.run-name "$run_name"
