import os
import sys
import traceback
from collections.abc import Mapping, MutableMapping
from importlib.metadata import entry_points
from pathlib import Path

import tyro
import warp as wp


def _is_wsl(
  *,
  env: Mapping[str, str] | None = None,
  osrelease_path: Path | str = "/proc/sys/kernel/osrelease",
) -> bool:
  """Return True when running inside Windows Subsystem for Linux (WSL)."""
  target_env = os.environ if env is None else env
  if target_env.get("WSL_DISTRO_NAME") or target_env.get("WSL_INTEROP"):
    return True
  try:
    osrelease = Path(osrelease_path).read_text(encoding="utf-8", errors="ignore")
  except OSError:
    return False
  return "microsoft" in osrelease.lower()


def _configure_linux_gl_env(
  *,
  platform: str | None = None,
  env: MutableMapping[str, str] | None = None,
  osrelease_path: Path | str = "/proc/sys/kernel/osrelease",
) -> None:
  """Configure Linux/WSL OpenGL environment defaults before importing MuJoCo.

  Defaults to ``MUJOCO_GL=egl`` on Linux for GPU-accelerated offscreen rendering
  (override with e.g. ``MUJOCO_GL=osmesa`` on clusters without EGL). On WSL2,
  Mesa >= 24.3 defaults window contexts to software ``llvmpipe`` unless
  ``GALLIUM_DRIVER=d3d12`` is set.
  """
  plat = sys.platform if platform is None else platform
  if not plat.startswith("linux"):
    return
  target_env = os.environ if env is None else env
  target_env.setdefault("MUJOCO_GL", "egl")
  if _is_wsl(env=target_env, osrelease_path=osrelease_path):
    target_env.setdefault("GALLIUM_DRIVER", "d3d12")


_configure_linux_gl_env()

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
  wp.config.log_level = wp.LOG_WARNING if quiet else wp.LOG_INFO


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
    except Exception:
      print(
        f"[WARN] Failed to load task package '{entry_point.name}' ({entry_point.value}):",
        file=sys.stderr,
      )
      traceback.print_exc(file=sys.stderr)


def _configure_mediapy() -> None:
  """Point mediapy at the bundled imageio-ffmpeg binary."""
  import imageio_ffmpeg
  import mediapy

  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


_configure_warp()
_configure_mediapy()
_import_registered_packages()
