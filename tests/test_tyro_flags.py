"""Test that all tyro.cli() calls include mjlab.TYRO_FLAGS."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SEARCH_DIRS = [ROOT / "src", ROOT / "scripts"]


def find_tyro_cli_calls() -> list[tuple[Path, int, bool]]:
  """Find all tyro.cli() calls and check if they have TYRO_FLAGS.

  Returns a list of (filepath, line_number, has_tyro_flags) tuples.
  """
  results = []

  for search_dir in SEARCH_DIRS:
    if not search_dir.exists():
      continue
    for filepath in search_dir.rglob("*.py"):
      try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
      except SyntaxError:
        continue

      for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
          continue

        # Check if this is a tyro.cli() call
        func = node.func
        is_tyro_cli = False

        if isinstance(func, ast.Attribute) and func.attr == "cli":
          # tyro.cli(...)
          if isinstance(func.value, ast.Name) and func.value.id == "tyro":
            is_tyro_cli = True
        elif isinstance(func, ast.Name) and func.id == "cli":
          # from tyro import cli; cli(...)
          is_tyro_cli = True

        if not is_tyro_cli:
          continue

        # Check if config= keyword argument contains TYRO_FLAGS
        has_tyro_flags = False
        for keyword in node.keywords:
          if keyword.arg == "config":
            config_source = ast.unparse(keyword.value)
            if "TYRO_FLAGS" in config_source:
              has_tyro_flags = True
              break

        results.append((filepath, node.lineno, has_tyro_flags))

  return results


def test_tyro_cli_calls_have_tyro_flags():
  """Ensure all tyro.cli() calls include config=mjlab.TYRO_FLAGS."""
  all_calls = find_tyro_cli_calls()
  assert all_calls, "No tyro.cli() calls found"

  violations = [
    (filepath, line) for filepath, line, has_flags in all_calls if not has_flags
  ]

  if violations:
    msg = "tyro.cli() calls missing config=mjlab.TYRO_FLAGS:\n"
    for filepath, line in violations:
      rel_path = filepath.relative_to(ROOT)
      msg += f"  {rel_path}:{line}\n"
    pytest.fail(msg)
