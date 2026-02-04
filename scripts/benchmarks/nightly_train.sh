#!/bin/bash
# Nightly training script for MJLab benchmarks
#
# This script clones mjlab fresh, runs the tracking benchmark, and generates a report.
# It is designed to be called by a systemd timer or cron job.
#
# Usage:
#   ./scripts/benchmarks/nightly_train.sh
#
# Environment variables:
#   CUDA_DEVICE: GPU device to use (default: 0)
#   WANDB_TAGS: Comma-separated tags for the run (default: nightly)
#   SKIP_TRAINING: Set to "1" to skip training and only generate report
#   SKIP_THROUGHPUT: Set to "1" to skip throughput benchmarking

set -euo pipefail

# Configuration
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WANDB_TAGS="${WANDB_TAGS:-(\"nightly\",)}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_THROUGHPUT="${SKIP_THROUGHPUT:-0}"

# Training configuration
TASK="Mjlab-Tracking-Flat-Unitree-G1"
NUM_ENVS=4096
MAX_ITERATIONS=6000
REGISTRY_NAME="rll_humanoid/wandb-registry-Motions/side_kick_test"

REPO_URL="git@github.com:mujocolab/mjlab.git"
GH_PAGES_BRANCH="gh-pages"
WORK_DIR="/tmp/mjlab-nightly-$$"
GH_PAGES_DIR="/tmp/mjlab-gh-pages-$$"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

clear_gpu() {
    local gpu_device="$1"
    log "Clearing GPU $gpu_device..."
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu_device" 2>/dev/null || true)
    if [[ -n "$gpu_pids" ]]; then
        for pid in $gpu_pids; do
            log "Killing process $pid on GPU $gpu_device"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 2  # Wait for processes to fully terminate
    fi
}

cleanup() {
    if [[ -d "$WORK_DIR" ]]; then
        log "Cleaning up work directory..."
        rm -rf "$WORK_DIR"
    fi
    if [[ -d "$GH_PAGES_DIR" ]]; then
        log "Cleaning up gh-pages clone..."
        rm -rf "$GH_PAGES_DIR"
    fi
}
trap cleanup EXIT

export GIT_SSH_COMMAND="ssh -i \"$HOME/.ssh/mjlab_nightly_ed25519\" \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new"

# Clone fresh copy of mjlab
log "Cloning mjlab..."
git clone "$REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"

log "Starting nightly benchmark run"
log "Task: $TASK"
log "GPU: $CUDA_DEVICE"
log "Commit: $(git rev-parse HEAD)"

# Run training
if [[ "$SKIP_TRAINING" != "1" ]]; then
    log "Starting training..."

    clear_gpu "$CUDA_DEVICE"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" uv run train "$TASK" \
        --env.scene.num-envs "$NUM_ENVS" \
        --agent.max-iterations "$MAX_ITERATIONS" \
        --registry-name "$REGISTRY_NAME" \
        --agent.wandb-tags "$WANDB_TAGS"

    log "Training completed"
else
    log "Skipping training (SKIP_TRAINING=1)"
fi

# Clone gh-pages branch (shallow clone for speed)
log "Cloning gh-pages branch..."
if git ls-remote --exit-code --heads origin "$GH_PAGES_BRANCH" > /dev/null 2>&1; then
    git clone --branch "$GH_PAGES_BRANCH" --depth 1 "$REPO_URL" "$GH_PAGES_DIR"
else
    # Create new gh-pages branch
    mkdir -p "$GH_PAGES_DIR"
    cd "$GH_PAGES_DIR"
    git init
    git remote add origin "$REPO_URL"
    git checkout -b "$GH_PAGES_BRANCH"
    cd "$WORK_DIR"
fi

# Copy cached data if exists
REPORT_DIR="$GH_PAGES_DIR/nightly"
mkdir -p "$REPORT_DIR"

# Run throughput benchmark
if [[ "$SKIP_THROUGHPUT" != "1" ]]; then
    log "Running throughput benchmark..."

    clear_gpu "$CUDA_DEVICE"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" uv run python scripts/benchmarks/measure_throughput.py \
        --num-envs "$NUM_ENVS" \
        --output-dir "$REPORT_DIR"
    log "Throughput benchmark completed"
else
    log "Skipping throughput benchmark (SKIP_THROUGHPUT=1)"
fi

# Generate report (uses cached data.json if present, only evaluates new runs)
log "Generating benchmark report..."
uv run python scripts/benchmarks/generate_report.py \
    --entity gcbc_researchers \
    --tag nightly \
    --output-dir "$REPORT_DIR"

log "Report generated"

# Commit and push
cd "$GH_PAGES_DIR"
git add -A
if git diff --staged --quiet; then
    log "No changes to commit"
else
    git commit -m "Update nightly tracking benchmark $(date '+%Y-%m-%d')"
    git push origin "$GH_PAGES_BRANCH" || log "Failed to push"
    log "Deployed to GitHub Pages"
fi

log "Nightly benchmark complete"
