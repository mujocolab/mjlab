"""Make the generated MuJoCo stubs platform-independent and reproducible.

``pybind11-stubgen`` introspects the installed ``mujoco`` package, producing two
kinds of machine-specific output:

1. The active OpenGL backend is baked in: ``mujoco.cgl`` on macOS,
   ``mujoco.glfw`` on Linux (or ``mujoco.egl`` / ``mujoco.osmesa`` depending on
   ``MUJOCO_GL``). mjlab does not type-check against the rendering backend, so we
   drop those stubs and the ``__init__.pyi`` imports referencing them.
2. Absolute install paths and live object reprs (memory addresses) are embedded
   as default values and ``# value = ...`` comments. These vary by machine and
   run, so we strip them.

What remains is a deterministic set of physics-binding stubs that is identical
on every platform, which lets CI regenerate them and fail if they drift from the
installed mujoco version.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

STUB_DIR = Path(__file__).parent / "mujoco"

# OpenGL context backends mujoco may expose, only one of which exists per
# platform / MUJOCO_GL setting.
BACKENDS = ("cgl", "egl", "glfw", "glx", "osmesa")

# Backend modules plus the rendering stubs that import them. None are needed to
# type-check mjlab, and all are platform-specific, so remove them entirely.
REMOVE = (*BACKENDS, "gl_context.pyi", "renderer.pyi", "rendering")

# Backend imports in __init__.pyi, e.g. "from mujoco.cgl import GLContext" and
# "from . import cgl", along with the names they contribute to __all__.
DROP_IMPORT = re.compile(
  rf"^(?:from mujoco\.(?:{'|'.join(BACKENDS)}) import GLContext"
  rf"|from \. import (?:{'|'.join(BACKENDS)}))$"
)
# Module-level constants in __init__.pyi that exist only on some platforms
# (macOS exposes is_rosetta / proc_translated for Rosetta detection; Linux does
# not). Remove them so the stubs match on every platform.
PLATFORM_ONLY = ("is_rosetta", "proc_translated")
PLATFORM_ONLY_DEF = re.compile(rf"^(?:{'|'.join(PLATFORM_ONLY)})\b")

DROP_FROM_ALL = ("GLContext", *BACKENDS, *PLATFORM_ONLY)

# _SYSTEM holds the host platform name ('Darwin' / 'Linux'); drop the value.
SYSTEM_DEFAULT = re.compile(r"^(_SYSTEM: str) = '\w+'$")

# A trailing "# value = ..." comment that captures a live object repr.
VALUE_COMMENT = re.compile(r" {2,}# value = .*$")

# A string constant whose default is an absolute install path.
PATH_DEFAULT = re.compile(
  r"^(\s*\w+: str) = '[^']*(?:site-packages|/Users/|/home/)[^']*'\s*$"
)


def remove_backend_stubs() -> None:
  for name in REMOVE:
    target = STUB_DIR / name
    if target.is_dir():
      shutil.rmtree(target)
    elif target.exists():
      target.unlink()


def patch_init() -> None:
  init = STUB_DIR / "__init__.pyi"
  out = []
  for line in init.read_text().splitlines():
    if DROP_IMPORT.match(line.strip()) or PLATFORM_ONLY_DEF.match(line):
      continue
    line = SYSTEM_DEFAULT.sub(r"\1", line)
    if line.startswith("__all__"):
      for name in DROP_FROM_ALL:
        line = line.replace(f"'{name}', ", "")
    out.append(line)
  init.write_text("\n".join(out) + "\n")


def sanitize_machine_specific() -> None:
  for path in STUB_DIR.rglob("*.pyi"):
    out = []
    for line in path.read_text().splitlines():
      if "# value = " in line and ("site-packages" in line or "0x" in line):
        line = VALUE_COMMENT.sub("", line)
      match = PATH_DEFAULT.match(line)
      if match:
        line = match.group(1)
      out.append(line)
    path.write_text("\n".join(out) + "\n")


def main() -> None:
  remove_backend_stubs()
  patch_init()
  sanitize_machine_specific()


if __name__ == "__main__":
  main()
