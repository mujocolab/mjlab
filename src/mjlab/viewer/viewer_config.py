import enum
from dataclasses import dataclass


@dataclass(kw_only=True)
class ViewerConfig:
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

  # Options that apply to every origin type.
  origin_type: OriginType = OriginType.AUTO

  height: int = 240
  width: int = 320

  enable_reflections: bool = True
  enable_shadows: bool = True

  geom_group: tuple[int, int, int, int, int, int] = (1, 1, 1, 0, 0, 0)
  """Which geom visualization groups the offscreen renderer draws."""

  site_group: tuple[int, int, int, int, int, int] = (1, 1, 1, 0, 0, 0)
  """Which site visualization groups the offscreen renderer draws."""

  env_idx: int = 0
  """Environment rendered as the primary env and tracked by the camera."""

  max_extra_envs: int = 2
  """Number of neighboring environments to render around env_idx."""

  reward_bar_max_terms: int = 20
  """Maximum number of reward terms shown in the Viser reward bar panel.

  Terms beyond this limit are dropped (with a warning). Raise it for
  environments with many reward terms."""

  # Camera placement.
  distance: float = 5.0
  elevation: float = -45.0
  azimuth: float = 90.0

  fovy: float | None = None
  """Vertical field of view in degrees. None keeps the model default."""

  lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Look-at target in the world frame. Only used by the AUTO and WORLD free
  cameras; the ASSET_ROOT and ASSET_BODY tracking cameras ignore it."""

  # Fields specific to the ASSET_ROOT and ASSET_BODY tracking cameras.
  entity_name: str | None = None
  """Entity tracked by the camera. Auto-detected if the scene has exactly one
  entity."""

  body_name: str | None = None
  """Body tracked by the camera. Required for ASSET_BODY."""
