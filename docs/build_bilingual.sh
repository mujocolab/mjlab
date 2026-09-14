#!/usr/bin/env bash
# 中英双语文档一键构建脚本:英文站输出 docs/_build,中文站输出 docs/_build/zh
# 用法: bash docs/build_bilingual.sh [额外的 sphinx-build 选项, 如 -E]
set -euo pipefail
cd "$(dirname "$0")/.." # 仓库根目录

uv run --group docs sphinx-build -j auto docs docs/_build "$@"
MJLAB_DOC_LANG=zh uv run --group docs sphinx-build -j auto -c docs docs/zh docs/_build/zh "$@"

echo "双语构建完成: docs/_build (英文) + docs/_build/zh (中文)"
