#!/usr/bin/env bash

# Generate type stubs for MuJoCo using pybind11-stubgen
# This helps suppress pyright errors for MuJoCo's C++ bindings

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing pybind11-stubgen if not already installed..."
uv pip install -q pybind11-stubgen

echo "Generating MuJoCo type stubs..."
uv run pybind11-stubgen mujoco -o "$SCRIPT_DIR" --ignore-all-errors
uv run pybind11-stubgen mujoco.viewer -o "$SCRIPT_DIR" --ignore-all-errors

echo "MuJoCo stubs generated successfully in $SCRIPT_DIR/mujoco/"
echo ""
echo "Make sure your pyproject.toml includes:"
echo "  [tool.pyright]"
echo "  stubPath = \"typings\""
