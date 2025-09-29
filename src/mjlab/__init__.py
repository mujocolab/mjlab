import os
from pathlib import Path

import warp as wp

MJLAB_SRC_PATH: Path = Path(__file__).parent


def configure_warp() -> None:
  """Configure Warp globally for mjlab."""
  wp.config.enable_backward = False

  # Keep warp verbose by default to show kernel compilation progress.
  # Override with MJLAB_WARP_QUIET=1 environment variable if needed.
  quiet = os.environ.get("MJLAB_WARP_QUIET", "").lower() in ("1", "true", "yes")
  wp.config.quiet = quiet


configure_warp()
