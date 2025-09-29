import os
from pathlib import Path

import warp as wp

MJLAB_SRC_PATH: Path = Path(__file__).parent


def configure_warp() -> None:
  """Configure Warp globally for mjlab."""
  wp.config.enable_backward = False

  # By default, we want to keep warp verbose especially when it is compiling kernels
  # so that users get feedback that something is happening.
  # However, we allow overriding this behavior with an environment variable.
  quiet = os.environ.get("MJLAB_WARP_QUIET", "").lower() in ("1", "true", "yes")
  wp.config.quiet = quiet


configure_warp()
