#!/usr/bin/env bash
set -euo pipefail

cd /home/ut/football_project/mjlab_soccer
exec env -u VIRTUAL_ENV uv run train \
  Mjlab-Velocity-Walk-KlavierReplica-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 30001 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name unitree_g1_flat_copied_model_seed42_30k_wandb
