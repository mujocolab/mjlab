#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
source_run="2026-08-19_16-50-17_DepthStudent_DirectLatent_v4_correct_teacher_domainrand"
source_checkpoint="model_9999.pt"
checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_depth_temporal_distillation/$source_run/$source_checkpoint"
run_name="DepthStudent_DeploymentRobustV2_mixed_rollout_seed42_resume9999_5k_wandb"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: student checkpoint does not exist: $checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR=/tmp/uv-cache
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run train \
  Mjlab-Velocity-Football-Depth-TemporalTeacher-DeploymentRobustV2-Distillation-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.student.cnn-cfg.freeze-coordinate-actor False \
  --agent.max-iterations 5000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
