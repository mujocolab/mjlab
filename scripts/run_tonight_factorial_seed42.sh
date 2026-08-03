#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
current_pid="91823"
current_run="$repo_dir/logs/rsl_rl/g1_velocity_football/2026-08-03_17-17-51_B1_A1R0_strict_IsaacLab_seed42_5k_wandb"

cd "$repo_dir"
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

echo "[$(date '+%F %T')] Waiting for A1R0 PID $current_pid"
while kill -0 "$current_pid" 2>/dev/null; do
  sleep 30
done

if [[ ! -f "$current_run/model_4999.pt" ]]; then
  echo "[$(date '+%F %T')] ERROR: A1R0 ended without model_4999.pt"
  exit 1
fi
echo "[$(date '+%F %T')] A1R0 complete"

run_experiment() {
  local task_id="$1"
  local run_name="$2"
  echo "[$(date '+%F %T')] Starting $run_name"
  .venv/bin/train "$task_id" \
    --env.scene.num-envs 4096 \
    --agent.seed 42 \
    --agent.max-iterations 5000 \
    --agent.save-interval 1000 \
    --agent.logger wandb \
    --agent.upload-model False \
    --agent.run-name "$run_name"
  echo "[$(date '+%F %T')] Completed $run_name"
}

run_experiment \
  Mjlab-Velocity-Football-A0R0-Flat-Unitree-G1 \
  A0R0_MLP_IsaacLab_seed42_5k_wandb

run_experiment \
  Mjlab-Velocity-Football-A0R1-Flat-Unitree-G1 \
  A0R1_MLP_ball_center_seed42_5k_wandb

echo "[$(date '+%F %T')] Tonight factorial queue complete"
