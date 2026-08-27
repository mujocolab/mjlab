#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR=/home/ut/football_project/mjlab_soccer
STATUS_LOG=/tmp/weekend_history_ablation_status.log
GPU_POLL_SECONDS=60

tasks=(
  "Mjlab-Velocity-Football-VisualMask-Flat-Unitree-G1"
  "Mjlab-Velocity-Football-Temporal-Flat-Unitree-G1"
)
labels=(
  "current_only"
  "history10_temporalcnn"
)
seeds=(42 43 44)

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
printf '%s GPU_AVAILABLE QUEUE_START\n' "$(date '+%F %T')" >>"$STATUS_LOG"

for seed in "${seeds[@]}"; do
  for index in "${!tasks[@]}"; do
    task=${tasks[$index]}
    label=${labels[$index]}
    run_name="E3_hist_ablation_${label}_seed${seed}"
    train_log="/tmp/${run_name}.log"

    printf '%s START run=%s task=%s seed=%s\n' \
      "$(date '+%F %T')" "$run_name" "$task" "$seed" >>"$STATUS_LOG"

    env -u VIRTUAL_ENV uv run train "$task" \
      --env.scene.num-envs 4096 \
      --agent.seed "$seed" \
      --agent.max-iterations 30000 \
      --agent.save-interval 1000 \
      --agent.logger tensorboard \
      --agent.upload-model False \
      --agent.run-name "$run_name" >"$train_log" 2>&1
    exit_code=$?

    printf '%s END run=%s exit_code=%s\n' \
      "$(date '+%F %T')" "$run_name" "$exit_code" >>"$STATUS_LOG"
    if [[ $exit_code -ne 0 ]]; then
      printf '%s RUN_FAILED_CONTINUING run=%s\n' \
        "$(date '+%F %T')" "$run_name" >>"$STATUS_LOG"
    fi
  done
done

printf '%s QUEUE_COMPLETE\n' "$(date '+%F %T')" >>"$STATUS_LOG"
