import abc
from typing import Dict, Optional

import mujoco

from mjlab.core import entity
from mjlab import MJLAB_ROOT_PATH

_XML = MJLAB_ROOT_PATH / "mjlab" / "entities" / "arenas" / "xmls" / "arena.xml"


class Arena(entity.Entity, abc.ABC):
  """An empty arena."""

  def __init__(self, assets: Optional[Dict[str, bytes]] = None):
    spec = mujoco.MjSpec.from_file(str(_XML), assets=assets)
    super().__init__(spec)

  @property
  @abc.abstractmethod
  def floor_geom(self) -> mujoco.MjsGeom:
    """The floor geometry."""
    raise NotImplementedError
