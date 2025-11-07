import os
from importlib.metadata import entry_points
from pathlib import Path

import warp as wp

__all__ = ["MJLAB_SRC_PATH"]

MJLAB_SRC_PATH: Path = Path(__file__).parent


def _configure_warp() -> None:
  """Configure Warp globally for mjlab."""
  wp.config.enable_backward = False

  # Keep warp verbose by default to show kernel compilation progress.
  # Override with MJLAB_WARP_QUIET=1 environment variable if needed.
  quiet = os.environ.get("MJLAB_WARP_QUIET", "0").lower() in ("1", "true", "yes")
  wp.config.quiet = quiet


def _import_registered_packages() -> None:
  """Auto-discover and import packages registered via entry points.

  Looks for packages registered under the 'mjlab.tasks' entry point group.
  Each discovered package is imported, which allows it to register custom
  environments with gymnasium.
  """
  mjlab_tasks = entry_points().select(group="mjlab.tasks")
  for entry_point in mjlab_tasks:
    try:
      entry_point.load()
    except Exception as e:
      print(f"[WARN] Failed to load task package {entry_point.name}: {e}")


_configure_warp()
_import_registered_packages()
