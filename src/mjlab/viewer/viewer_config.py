import enum
from dataclasses import dataclass


@dataclass
class ViewerConfig:
  lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
  distance: float = 5.0
  elevation: float = -45.0
  azimuth: float = 90.0

  class OriginType(enum.Enum):
    """The frame in which the camera position and target are defined."""

    WORLD = enum.auto()
    """The origin of the world."""
    ASSET_ROOT = enum.auto()
    """The center of the asset defined by asset_name."""
    ASSET_BODY = enum.auto()
    """The center of the body defined by body_name in asset defined by asset_name."""

  origin_type: OriginType = OriginType.WORLD
  asset_name: str | None = None
  body_name: str | None = None
