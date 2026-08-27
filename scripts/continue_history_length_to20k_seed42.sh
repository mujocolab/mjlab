#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=/home/ut/football_project/mjlab_soccer
STATUS_LOG=/tmp/history_length_ablation_seed42_status.log
RUNS=(
  "2026-08-01_22-44-37_E3_history_length_history5_seed42_5k"
  "2026-08-01_23-25-33_E3_history_length_history10_seed42_5k"
  "2026-08-02_00-11-39_E3_history_length_history20_seed42_5k"
)
TASKS=(
  "Mjlab-Velocity-Football-Temporal-History5-Flat-Unitree-G1"
  "Mjlab-Velocity-Football-Temporal-Flat-Unitree-G1"
  "Mjlab-Velocity-Football-Temporal-History20-Flat-Unitree-G1"
)
LABELS=(history5 history10 history20)
EXPECTED_HISTORY=(5 10 20)
NUM_ENVS=1024

cd "$PROJECT_DIR" || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export XDG_CACHE_HOME=/tmp/xdg-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache

for index in "${!RUNS[@]}"; do
  source_run=${RUNS[$index]}
  task=${TASKS[$index]}
  label=${LABELS[$index]}
  expected_history=${EXPECTED_HISTORY[$index]}
  checkpoint="$PROJECT_DIR/logs/rsl_rl/g1_velocity_football/$source_run/model_4999.pt"
  if [[ ! -f "$checkpoint" ]]; then
    printf '%s QUEUE_STOP label=%s missing_checkpoint=%s\n' \
      "$(date '+%F %T')" "$label" "$checkpoint" >>"$STATUS_LOG"
    exit 1
  fi
  run_name="E3_history_length_${label}_seed42_20k"
  train_log="/tmp/${run_name}.log"
  printf '%s RESUME_START label=%s task=%s history=%s num_envs=%s source=%s target=20000\n' \
    "$(date '+%F %T')" "$label" "$task" "$expected_history" \
    "$NUM_ENVS" "$source_run" >>"$STATUS_LOG"
  # RSL-RL adds the loaded checkpoint iteration to this value.  The 5k
  # source checkpoint is iteration 4999, so 15000 new iterations ends at
  # iteration 19999 (20k total).
  env -u VIRTUAL_ENV uv run train "$task" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.resume True \
    --agent.load-run "$source_run" \
    --agent.load-checkpoint 'model_4999.pt' \
    --agent.max-iterations 15000 \
    --agent.save-interval 1000 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.run-name "$run_name" >"$train_log" 2>&1
  exit_code=$?
  printf '%s RESUME_END label=%s exit_code=%s\n' \
    "$(date '+%F %T')" "$label" "$exit_code" >>"$STATUS_LOG"
  if [[ $exit_code -ne 0 ]]; then
    exit "$exit_code"
  fi
done

printf '%s ALL_20K_COMPLETE\n' "$(date '+%F %T')" >>"$STATUS_LOG"
