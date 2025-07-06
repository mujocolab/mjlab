from typing import Optional
import mujoco
from mjlab.core import entity
from mjlab import MJLAB_ROOT_PATH
from dataclasses import dataclass

from mjlab.entities.terrains import editors

_XML = MJLAB_ROOT_PATH / "mjlab" / "entities" / "terrains" / "xmls" / "terrain.xml"


@dataclass(frozen=True)
class TerrainConfig:
  terrain: editors.TerrainEditor
  skybox: Optional[editors.Skybox] = None

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    if self.skybox is not None:
      self.skybox.edit_spec(spec)
    self.terrain.edit_spec(spec)


class Terrain(entity.Entity):
  def __init__(self, config: TerrainConfig | None = None):
    self._spec = mujoco.MjSpec.from_file(str(_XML))
    self._config = config
    if config is not None:
      config.edit_spec(self._spec)

    self._floor_geom = self.spec.geom(config.terrain.name)

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def floor_geom(self) -> mujoco.MjsGeom:
    return self._floor_geom
