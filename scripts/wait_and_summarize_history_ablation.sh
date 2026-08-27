#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR=/home/ut/football_project/mjlab_soccer
STATUS_LOG=/tmp/weekend_history_ablation_status.log
SUMMARY_LOG=/tmp/weekend_history_ablation_summary.log

while ! grep -q "QUEUE_COMPLETE" "$STATUS_LOG" 2>/dev/null; do
  sleep 60
done

cd "$PROJECT_DIR" || exit 1
export UV_CACHE_DIR=/tmp/uv-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

uv run python scripts/summarize_history_ablation.py >"$SUMMARY_LOG" 2>&1
