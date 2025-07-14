from pathlib import Path

MJLAB_SRC_PATH = Path(__file__).parent
MJLAB_ROOT_PATH = MJLAB_SRC_PATH.parent

__all__ = (
  "MJLAB_SRC_PATH",
  "MJLAB_ROOT_PATH",
)