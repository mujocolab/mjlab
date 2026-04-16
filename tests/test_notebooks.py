"""Validate that notebook cells import only symbols that exist in mjlab.

The cartpole tutorial notebook writes Python files via ``%%writefile`` cells
and imports directly from ``mjlab`` in others. When a public API is renamed
or removed, those references silently rot until a user runs the notebook.
This test parses each code cell, finds ``from mjlab... import X`` statements,
and verifies that ``X`` is actually importable.

Modules that the notebook itself creates (via ``%%writefile`` into the mjlab
source tree) are skipped, since they don't exist until the notebook runs.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"

# Match "%%writefile /content/mjlab/src/mjlab/foo/bar.py" style targets that
# land inside the mjlab package.
_WRITEFILE_MJLAB_RE = re.compile(
  r"^\s*%%writefile\s+\S*?/src/(mjlab(?:/[\w/]+)+)\.py\s*$"
)


def _extract_python_source(cell_source: str) -> str | None:
  """Return Python source for a cell, or None if it has no Python to parse.

  Handles ``%%writefile path.py`` cells by returning their body (only when the
  target is a ``.py`` file; XML/YAML/etc. writefiles are skipped). Strips
  shell escapes (``!...``) and line magics (``%...``).
  """
  lines = cell_source.splitlines()
  if lines and lines[0].lstrip().startswith("%%writefile"):
    target = lines[0].split(maxsplit=1)[1] if len(lines[0].split()) > 1 else ""
    if not target.endswith(".py"):
      return None
    return "\n".join(lines[1:])
  # Skip other cell magics (%%bash, %%html, etc.) entirely.
  if lines and lines[0].lstrip().startswith("%%"):
    return None
  kept = [ln for ln in lines if not ln.lstrip().startswith(("!", "%"))]
  return "\n".join(kept)


def _iter_mjlab_imports(source: str):
  """Yield (module, name) for every ``from mjlab... import name`` in source."""
  tree = ast.parse(source)
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module:
      if node.module == "mjlab" or node.module.startswith("mjlab."):
        for alias in node.names:
          yield node.module, alias.name


def _collect_written_modules(nb: dict) -> set[str]:
  """Return the set of mjlab modules the notebook creates via %%writefile."""
  written: set[str] = set()
  for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
      continue
    source = "".join(cell.get("source", []))
    first_line = source.splitlines()[0] if source else ""
    match = _WRITEFILE_MJLAB_RE.match(first_line)
    if match:
      written.add(match.group(1).replace("/", "."))
  return written


def _collect_notebook_imports(path: Path):
  """Return (imports, written_modules) where imports is a list of
  (cell_index, module, name) tuples for every ``from mjlab... import`` in the
  notebook, and written_modules is the set of mjlab modules the notebook
  creates itself.
  """
  nb = json.loads(path.read_text())
  written = _collect_written_modules(nb)
  imports = []
  for idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
      continue
    source = "".join(cell.get("source", []))
    python_src = _extract_python_source(source)
    if not python_src or not python_src.strip():
      continue
    try:
      cell_imports = list(_iter_mjlab_imports(python_src))
    except SyntaxError as e:
      pytest.fail(f"Cell {idx} in {path.name} has invalid Python syntax: {e}")
    for module, name in cell_imports:
      imports.append((idx, module, name))
  return imports, written


@pytest.mark.parametrize("notebook_path", sorted(NOTEBOOKS_DIR.glob("*.ipynb")))
def test_notebook_mjlab_imports_resolve(notebook_path: Path) -> None:
  """Every ``from mjlab... import X`` in a notebook must resolve."""
  imports, written_modules = _collect_notebook_imports(notebook_path)
  failures = []
  for cell_idx, module, name in imports:
    # Skip modules the notebook writes itself; they don't exist until runtime.
    if module in written_modules or any(
      module.startswith(w + ".") for w in written_modules
    ):
      continue
    try:
      mod = importlib.import_module(module)
    except ImportError as e:
      failures.append(
        f"cell {cell_idx}: `from {module} import {name}` failed to import module: {e}"
      )
      continue
    if not hasattr(mod, name):
      failures.append(
        f"cell {cell_idx}: `from {module} import {name}` — "
        f"'{name}' is not exposed by '{module}'"
      )
  if failures:
    pytest.fail(
      f"Stale mjlab imports in {notebook_path.name}:\n  " + "\n  ".join(failures)
    )
