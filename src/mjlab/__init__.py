import os
from pathlib import Path

import warp as wp

MJLAB_SRC_PATH: Path = Path(__file__).parent
MJLAB_ROOT_PATH: Path = MJLAB_SRC_PATH.parent


def configure_warp() -> None:
  """Configure Warp globally for mjlab."""
  wp.config.enable_backward = False

  quiet = os.environ.get("MJLAB_WARP_VERBOSE", "").lower() not in ("1", "true", "yes")
  wp.config.quiet = quiet


configure_warp()
