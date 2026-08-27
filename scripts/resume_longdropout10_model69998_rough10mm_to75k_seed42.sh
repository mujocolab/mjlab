#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
source_run="2026-08-12_10-05-41_B1_A1R0_longdropout10_rewardfade05_legacycurriculum_seed42_resume50k_to70k_wandb"
source_checkpoint="model_69998.pt"
run_name="B1_A1R0_longdropout10_roughcurriculum10mm_seed42_resume70k_to75k_wandb"
checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football/$source_run/$source_checkpoint"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing resume checkpoint: $checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

uv run train \
  Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-LegacyCurriculum-Rough10mm-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --env.commands.twist.ranges.lin-vel-x=-0.5,2.0 \
  --env.commands.twist.ranges.lin-vel-y=-0.5,0.5 \
  --env.commands.twist.ranges.ang-vel-z=-1.0,1.0 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.max-iterations 5002 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.upload-model False \
  --agent.run-name "$run_name"
