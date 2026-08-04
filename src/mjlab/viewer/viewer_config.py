import enum
from dataclasses import dataclass

from mjlab.entity import Entity


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
    """Track the body defined by body_name in the asset defined by entity_name."""

  # General viewer config
  origin_type: OriginType = OriginType.AUTO
  enable_reflections: bool = True
  enable_shadows: bool = True
  height: int = 240
  width: int = 320

  reward_bar_max_terms: int = 20
  """Maximum number of reward terms shown in the Viser reward bar panel.

  Terms beyond this limit are dropped (with a warning). Raise it for
  environments with many reward terms."""

  geom_group: tuple[int, int, int, int, int, int] = (1, 1, 1, 0, 0, 0)
  site_group: tuple[int, int, int, int, int, int] = (1, 1, 1, 0, 0, 0)

  # Relative camera position to lookat
  elevation: float = -45.0
  azimuth: float = 90.0
  distance: float = 5.0
  fovy: float | None = None

  lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Look-at target in the world frame. This target is only used by AUTO and WORLD free cameras.
    The tracking cameras used for ASSET_ROOT or ASSET_BODY **IGNORE** the 'lookat' field."""

  # TODO(): Defining the total number of rendered environments would be better but this change
  #  would break the existing API.
  max_extra_envs: int = 2
  """The number of additional rendered environments. The total number of environments
    is 'max_extra_envs' + 1. The additional environments are selected to be close to the 
    camera viewpoint. 
    
    For WORLD and AUTO, the selected environments are closest to 'lookat'.
    For ASSET_ROOT and ASSET_BODY, the selected environments are closest to the 'env_idx'
    """

  # ASSET_ROOT and ASSET_BODY specific fields
  env_idx: int = 0
  """The id of the environment that is being tracked by the camera."""

  entity_name: str | None = None
  """The name of the entity that is being tracked by the camera."""

  body_name: str | None = None
  """The name of the body that is being tracked by the camera."""


def get_camera_body_id(
  origin_type: ViewerConfig.OriginType,
  body_name: str | None,
  entity_name: str | None,
  entities: dict[str, Entity],
) -> int:
  """Infer the body id for the different origin types."""
  if origin_type in (ViewerConfig.OriginType.AUTO, ViewerConfig.OriginType.WORLD):
    return -1

  # Infer the entity that is used to read the body id
  entity_i: Entity

  if entity_name is not None:
    entity_i = entities[entity_name]

  elif len(entities) == 1:
    # Auto-detect if only one entity exists.
    entity_i = list(entities.values())[0]

  else:
    msg = (
      f"Cannot infer entity. The entity name is not specified and the scene contains "
      f"multiple entities (n = {len(entities)}).\n\nEntities: {list(entities.keys())}"
    )
    raise ValueError(msg)

  # Read the body-id from the entity
  if origin_type == ViewerConfig.OriginType.ASSET_ROOT:
    return entity_i.indexing.root_body_id

  elif origin_type == ViewerConfig.OriginType.ASSET_BODY:
    if not body_name:
      msg = f"For OriginType.ASSET_BODY the 'body_name' has to be specified, got '{body_name}'"
      raise ValueError(msg)

    if body_name not in entity_i.body_names:
      msg = f"Body '{body_name}' not found in asset '{entity_name}'"
      raise ValueError(msg)

    # TODO(): Should len(body_id_list) > 1 raise an error?
    body_id_list, _ = entity_i.find_bodies(body_name)
    return entity_i.indexing.bodies[body_id_list[0]].id

  msg = f"Unknown ViewerConfig.OriginType = {origin_type}"
  raise ValueError(msg)
