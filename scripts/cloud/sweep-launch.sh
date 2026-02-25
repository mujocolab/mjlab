#!/usr/bin/env bash
# Launch a W&B sweep via SkyPilot.
#
# Provisions a single multi-GPU cluster and runs one sweep agent per GPU
# using SkyPilot's job queue. Each agent pulls hyperparameters from the
# W&B sweep controller independently.
#
# Usage:
#   ./scripts/cloud/sweep-launch.sh [GPUS [CLOUD]]
#
# Examples:
#   ./scripts/cloud/sweep-launch.sh A100:4          # 4 agents, default cloud
#   ./scripts/cloud/sweep-launch.sh A100:8 gcp      # 8 agents on GCP
#   ./scripts/cloud/sweep-launch.sh A100:8 lambda   # 8 agents on Lambda
#   ./scripts/cloud/sweep-launch.sh                 # defaults to A100:4

set -euo pipefail

GPUS="${1:-A100:4}"
CLOUD="${2:-}"
GPU_TYPE="${GPUS%%:*}"
NUM_AGENTS="${GPUS##*:}"
CLUSTER_NAME="mjlab-sweep"

echo "Creating W&B sweep..."
SWEEP_ID=$(uv run wandb sweep scripts/cloud/sweep.yaml 2>&1 | grep "wandb agent" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
  echo "Failed to create sweep."
  exit 1
fi

echo "Sweep created: $SWEEP_ID"
echo "Provisioning $GPUS cluster..."

# Provision the cluster and run setup (no run section in this YAML).
CLOUD_FLAG=${CLOUD:+--cloud "$CLOUD"}
sky launch scripts/cloud/sweep-cluster.yaml \
  -c "$CLUSTER_NAME" \
  --gpus "$GPUS" \
  ${CLOUD_FLAG} \
  -y --retry-until-up

echo "Submitting $NUM_AGENTS agents to job queue..."

for i in $(seq 1 "$NUM_AGENTS"); do
  echo "  Agent $i/$NUM_AGENTS"
  sky exec "$CLUSTER_NAME" \
    --gpus "${GPU_TYPE}:1" \
    --env "SWEEP_ID=$SWEEP_ID" \
    -d \
    scripts/cloud/sweep-agent.yaml
done

echo ""
echo "All agents launched. Monitor at:"
echo "  sky queue $CLUSTER_NAME"
echo "  sky logs $CLUSTER_NAME <JOB_ID>"
echo "  W&B dashboard: https://wandb.ai/$SWEEP_ID"
echo ""
echo "When done: sky down $CLUSTER_NAME"
