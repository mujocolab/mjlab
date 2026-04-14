"""Shared helpers for CLI entrypoints."""

import sys


def maybe_print_top_level_help(prog: str) -> None:
  """Print a top-level usage message and exit if argv is bare -h/--help.

  The ``train`` and ``play`` entrypoints use a two-stage tyro parse with
  ``add_help=False`` on the first stage, so ``prog --help`` otherwise
  produces no output. This handles that case explicitly.
  """
  if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
    print(f"usage: {prog} <TASK> [OPTIONS]")
    print()
    print(f"Run '{prog} <TASK> --help' for task-specific options.")
    print("Run 'uv run list-envs' to list available tasks.")
    sys.exit(0)
