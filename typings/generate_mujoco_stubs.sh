#!/usr/bin/env bash

# Generate type stubs for MuJoCo using pybind11-stubgen.
# This helps suppress pyright/ty errors for MuJoCo's C++ bindings.
#
# The output is post-processed by postprocess_stubs.py to make it
# platform-independent (it drops the OpenGL backend stubs, which differ between
# macOS and Linux). This keeps the committed stubs identical on every platform,
# so CI can regenerate them and fail if they drift from the installed mujoco
# version. pybind11-stubgen is pinned so its output is reproducible.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --no-sync runs in the already-synced project environment (so mujoco is
# importable) without disturbing it; the conflicting cpu/cu128 extras mean a
# plain "uv run" would re-sync and churn the environment.
STUBGEN_VERSION="2.5.5"
STUBGEN=(uv run --no-sync --with "pybind11-stubgen==${STUBGEN_VERSION}" pybind11-stubgen)

echo "Generating MuJoCo type stubs..."
"${STUBGEN[@]}" mujoco -o "$SCRIPT_DIR" --ignore-all-errors

# mujoco.viewer pulls in GLFW, which may not import on a headless machine.
# Skip it in that case rather than failing; the committed viewer stub is kept.
if ! "${STUBGEN[@]}" mujoco.viewer -o "$SCRIPT_DIR" --ignore-all-errors; then
  echo "Warning: could not generate mujoco.viewer stubs (no GLFW?). Skipping."
fi

echo "Post-processing stubs to make them platform-independent..."
uv run --no-sync python "$SCRIPT_DIR/postprocess_stubs.py"

echo "MuJoCo stubs generated successfully in $SCRIPT_DIR/mujoco/"
