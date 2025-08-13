from mjlab.entities import entity
from mjlab.entities.terrains.terrain_config import TerrainCfg
from mjlab.utils.spec_editor import spec_editor as common_editors
from mjlab.entities.indexing import EntityIndexing
import mujoco_warp as mjwarp


class Terrain(entity.Entity):
  cfg: TerrainCfg

  def __init__(self, terrain_cfg: TerrainCfg):
    super().__init__(terrain_cfg)

  def reset(self, env_ids: entity.Sequence[int] | None = None):
    pass

  def write_data_to_sim(self) -> None:
    pass

  def update(self, dt: float) -> None:
    pass

  def initialize(
    self,
    indexing: EntityIndexing,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    pass

  # Private methods.

  def _configure_spec(self) -> None:
    super()._configure_spec()
    for geom in self.cfg.geoms:
      common_editors.GeomEditor(geom).edit_spec(self._spec)
    for col in self.cfg.collisions:
      common_editors.CollisionEditor(col).edit_spec(self._spec)
