import enum
from dataclasses import dataclass


@dataclass
class ViewerConfig:
  lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
  distance: float = 5.0
  fovy: float | None = None
  elevation: float = -45.0
  azimuth: float = 90.0

  class OriginType(enum.Enum):
    """The frame in which the camera position and target are defined."""

    AUTO = enum.auto()
    """Track the first non-fixed body, or fall back to a free camera."""
    WORLD = enum.auto()
    """Free camera at the configured lookat point."""
    ASSET_ROOT = enum.auto()
    """Track the root body of the asset defined by entity_name."""
    ASSET_BODY = enum.auto()
    """Track the body defined by body_name in the asset defined by
    entity_name."""

  origin_type: OriginType = OriginType.AUTO
  entity_name: str | None = None
  body_name: str | None = None
  env_idx: int = 0
  max_extra_envs: int = 2
  """Number of neighboring environments to render around ``env_idx``."""
  enable_reflections: bool = True
  enable_shadows: bool = True
  height: int = 240
  width: int = 320
