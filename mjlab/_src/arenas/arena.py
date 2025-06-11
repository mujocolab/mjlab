import abc
from typing import Dict, Optional

import mujoco

from mjlab._src import ROOT_PATH, entity

_XML = ROOT_PATH / "arenas" / "xmls" / "arena.xml"


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
