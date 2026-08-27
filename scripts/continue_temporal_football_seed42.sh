#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR=/home/ut/football_project/mjlab_soccer
STATUS_LOG=/tmp/temporal_two_stage_seed42_status.log
FOOTBALL_LOG=/tmp/temporal_football_seed42.log
WALK_RUN_DIR=logs/rsl_rl/g1_velocity_football_pretrain/2026-07-31_20-15-24_TemporalCNN_Walk_hist10_seed42_15k
WALK_CHECKPOINT="$PROJECT_DIR/$WALK_RUN_DIR/model_14999.pt"
FOOTBALL_RUN_NAME=TemporalCNN_Football_hist10_visualmask_seed42_20k

cd "$PROJECT_DIR" || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export XDG_CACHE_HOME=/tmp/xdg-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache

if [[ ! -f "$WALK_CHECKPOINT" ]]; then
  printf '%s QUEUE_STOP missing_checkpoint=%s\n' \
    "$(date '+%F %T')" "$WALK_CHECKPOINT" >>"$STATUS_LOG"
  exit 1
fi

printf '%s FOOTBALL_START run=%s iterations=20000 checkpoint=%s\n' \
  "$(date '+%F %T')" "$FOOTBALL_RUN_NAME" "$WALK_CHECKPOINT" >>"$STATUS_LOG"
env -u VIRTUAL_ENV uv run train \
  Mjlab-Velocity-Football-Temporal-Flat-Unitree-G1 \
  --pretrained-checkpoint "$WALK_CHECKPOINT" \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 20000 \
  --agent.save-interval 1000 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --agent.run-name "$FOOTBALL_RUN_NAME" >"$FOOTBALL_LOG" 2>&1
exit_code=$?
printf '%s FOOTBALL_END exit_code=%s\n' \
  "$(date '+%F %T')" "$exit_code" >>"$STATUS_LOG"
if [[ $exit_code -eq 0 ]]; then
  printf '%s QUEUE_COMPLETE\n' "$(date '+%F %T')" >>"$STATUS_LOG"
fi
exit "$exit_code"
