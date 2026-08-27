#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR=/home/ut/football_project/mjlab_soccer
STATUS_LOG=/tmp/history_length_ablation_seed42_status.log
WALK_CHECKPOINT="$PROJECT_DIR/logs/rsl_rl/g1_velocity_football_pretrain/2026-07-31_20-15-24_TemporalCNN_Walk_hist10_seed42_15k/model_14999.pt"
TASKS=(
  "Mjlab-Velocity-Football-Temporal-History5-Flat-Unitree-G1"
  "Mjlab-Velocity-Football-Temporal-Flat-Unitree-G1"
  "Mjlab-Velocity-Football-Temporal-History20-Flat-Unitree-G1"
)
LABELS=(history5 history10 history20)

cd "$PROJECT_DIR" || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export XDG_CACHE_HOME=/tmp/xdg-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache

if [[ ! -f "$WALK_CHECKPOINT" ]]; then
  printf '%s STOP missing_checkpoint=%s\n' \
    "$(date '+%F %T')" "$WALK_CHECKPOINT" >>"$STATUS_LOG"
  exit 1
fi

for index in "${!TASKS[@]}"; do
  task=${TASKS[$index]}
  label=${LABELS[$index]}
  run_name="E3_history_length_${label}_seed42_5k"
  train_log="/tmp/${run_name}.log"
  printf '%s START label=%s task=%s\n' \
    "$(date '+%F %T')" "$label" >>"$STATUS_LOG"
  env -u VIRTUAL_ENV uv run train "$task" \
    --pretrained-checkpoint "$WALK_CHECKPOINT" \
    --env.scene.num-envs 1024 \
    --agent.seed 42 \
    --agent.max-iterations 5000 \
    --agent.save-interval 1000 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.run-name "$run_name" >"$train_log" 2>&1
  exit_code=$?
  printf '%s END label=%s exit_code=%s\n' \
    "$(date '+%F %T')" "$label" "$exit_code" >>"$STATUS_LOG"
  if [[ $exit_code -ne 0 ]]; then
    exit "$exit_code"
  fi
done

printf '%s COMPLETE\n' "$(date '+%F %T')" >>"$STATUS_LOG"
