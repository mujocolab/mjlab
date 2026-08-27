#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
walk_checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_pretrain/2026-07-23_18-17-07/model_16000.pt"

cd "$repo_dir"

if [[ ! -f "$walk_checkpoint" ]]; then
  echo "ERROR: missing walking checkpoint: $walk_checkpoint" >&2
  exit 1
fi

export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

exec env \
  -u VIRTUAL_ENV \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u http_proxy \
  -u https_proxy \
  -u ALL_PROXY \
  -u all_proxy \
  WANDB_MODE=online \
  WANDB_INIT_TIMEOUT=180 \
  "$repo_dir/.venv/bin/train" \
  Mjlab-Velocity-Football-A1R0-Flat-Unitree-G1 \
  --pretrained-checkpoint "$walk_checkpoint" \
  --env.scene.num-envs 4096 \
  --env.commands.twist.heading-command False \
  --env.commands.twist.rel-heading-envs 0.0 \
  --env.commands.twist.ranges.lin-vel-x 0.1,1.0 \
  --env.commands.twist.ranges.lin-vel-y 0.0,0.0 \
  --env.commands.twist.ranges.ang-vel-z=-1.0,1.0 \
  --env.commands.twist.ranges.heading None \
  --env.rewards.track-ball-lin-vel-xy-exp.weight 4.0 \
  --env.rewards.track-ball-lin-vel-xy-exp.params.std 0.5 \
  --env.rewards.track-linear-velocity.weight 0.0 \
  --env.rewards.track-angular-velocity.weight 1.5 \
  --env.rewards.track-angular-velocity.params.std 0.5 \
  --env.rewards.track-ball-relative-vel-xy-exp.weight 0.0 \
  --env.rewards.track-ball-relative-pos-xy-exp.weight 0.0 \
  --env.rewards.ball-front-control.weight 1.0 \
  --env.rewards.ball-outside-control-zone.weight -1.0 \
  --agent.upload-model False \
  --agent.seed 42 \
  --agent.max-iterations 20000 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.run-name B1_A1R0_history10_forward_turn_ballvel4_ang1p5_seed42_from_walk16000_football20k_wandb
