#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
teacher_run="2026-08-14_11-44-01_B1_A1R0_longdropout10_isaac_actor_dr_flat_seed42_from_walk16000_to50k_wandb"
teacher_checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football/$teacher_run/model_49999.pt"
run_name="DepthStudent_MountRangeVisualDR_frozenMLP_teacherRollout_alpha025_noDelay_noLongDrop_pixel5_from_B1Teacher49999_seed42_10k_wandb"

if [[ ! -f "$teacher_checkpoint" ]]; then
  echo "ERROR: missing coordinate Teacher: $teacher_checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR=/tmp/uv-cache
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run train \
  Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeVisualDR-FrozenMLP-Distillation-Flat-Unitree-G1 \
  --pretrained-checkpoint "$teacher_checkpoint" \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.student.cnn-cfg.freeze-coordinate-actor True \
  --agent.algorithm.rollout-policy teacher \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
