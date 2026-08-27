#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
cd "$repo_dir"

uv run train Mjlab-Velocity-Football-Depth-Asymmetric-Flat-Unitree-G1 \
  --env.scene.num-envs 512 \
  --agent.seed 42 \
  --agent.max-iterations 30000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab-soccer \
  --agent.run-name DepthAsymmetric_V0_80x60_seed42
