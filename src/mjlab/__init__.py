import os
from importlib.metadata import entry_points
from pathlib import Path

import tyro
import warp as wp

MJLAB_SRC_PATH: Path = Path(__file__).parent

TYRO_FLAGS = (
  # Don't let users switch between types in unions. This produces a simpler CLI
  # with flatter helptext, at the cost of some flexibility. Type changes can
  # just be done in code.
  tyro.conf.AvoidSubcommands,
  # Disable automatic flag conversion (e.g., use `--flag False` instead of
  # `--no-flag` for booleans).
  tyro.conf.FlagConversionOff,
  # Use Python syntax for collections: --tuple (1,2,3) instead of --tuple 1 2 3.
  # Helps with wandb sweep compatibility: https://brentyi.github.io/tyro/wandb_sweeps/
  tyro.conf.UsePythonSyntaxForLiteralCollections,
)


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
