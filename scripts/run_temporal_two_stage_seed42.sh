#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR=/home/ut/football_project/mjlab_soccer
STATUS_LOG=/tmp/temporal_two_stage_seed42_status.log
WALK_LOG=/tmp/temporal_walk_seed42.log
FOOTBALL_LOG=/tmp/temporal_football_seed42.log
GPU_POLL_SECONDS=60
WALK_RUN_NAME=TemporalCNN_Walk_hist10_seed42_15k
FOOTBALL_RUN_NAME=TemporalCNN_Football_hist10_visualmask_seed42_20k

cd "$PROJECT_DIR" || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export XDG_CACHE_HOME=/tmp/xdg-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache

printf '%s QUEUE_WAITING_FOR_GPU\n' "$(date '+%F %T')" >>"$STATUS_LOG"
while true; do
  gpu_processes=$(nvidia-smi \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null) || {
    printf '%s GPU_QUERY_FAILED retry=%ss\n' \
      "$(date '+%F %T')" "$GPU_POLL_SECONDS" >>"$STATUS_LOG"
    sleep "$GPU_POLL_SECONDS"
    continue
  }
  if [[ -z "${gpu_processes//[[:space:]]/}" ]]; then
    sleep 30
    confirm_processes=$(nvidia-smi \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null) || continue
    if [[ -z "${confirm_processes//[[:space:]]/}" ]]; then
      break
    fi
  fi
  sleep "$GPU_POLL_SECONDS"
done

printf '%s WALK_START run=%s iterations=15000 seed=42\n' \
  "$(date '+%F %T')" "$WALK_RUN_NAME" >>"$STATUS_LOG"
env -u VIRTUAL_ENV uv run train \
  Mjlab-Velocity-Football-Temporal-Pretrain-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 15000 \
  --agent.save-interval 1000 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --agent.run-name "$WALK_RUN_NAME" >"$WALK_LOG" 2>&1
walk_exit_code=$?
printf '%s WALK_END exit_code=%s\n' \
  "$(date '+%F %T')" "$walk_exit_code" >>"$STATUS_LOG"
if [[ $walk_exit_code -ne 0 ]]; then
  printf '%s QUEUE_STOP stage=walk\n' "$(date '+%F %T')" >>"$STATUS_LOG"
  exit "$walk_exit_code"
fi

walk_dirs=(
  logs/rsl_rl/g1_velocity_football_pretrain/*_"$WALK_RUN_NAME"
)
walk_run_dir=${walk_dirs[-1]}
walk_checkpoint=""
for candidate in "$walk_run_dir/model_15000.pt" "$walk_run_dir/model_14999.pt"; do
  if [[ -f "$candidate" ]]; then
    walk_checkpoint="$candidate"
    break
  fi
done
if [[ -z "$walk_checkpoint" ]]; then
  printf '%s QUEUE_STOP missing_checkpoint=%s\n' \
    "$(date '+%F %T')" "$walk_checkpoint" >>"$STATUS_LOG"
  exit 1
fi

printf '%s FOOTBALL_START run=%s iterations=20000 checkpoint=%s\n' \
  "$(date '+%F %T')" "$FOOTBALL_RUN_NAME" "$walk_checkpoint" >>"$STATUS_LOG"
env -u VIRTUAL_ENV uv run train \
  Mjlab-Velocity-Football-Temporal-Flat-Unitree-G1 \
  --pretrained-checkpoint "$walk_checkpoint" \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 20000 \
  --agent.save-interval 1000 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --agent.run-name "$FOOTBALL_RUN_NAME" >"$FOOTBALL_LOG" 2>&1
football_exit_code=$?
printf '%s FOOTBALL_END exit_code=%s\n' \
  "$(date '+%F %T')" "$football_exit_code" >>"$STATUS_LOG"
if [[ $football_exit_code -ne 0 ]]; then
  printf '%s QUEUE_STOP stage=football\n' "$(date '+%F %T')" >>"$STATUS_LOG"
  exit "$football_exit_code"
fi

printf '%s QUEUE_COMPLETE\n' "$(date '+%F %T')" >>"$STATUS_LOG"
