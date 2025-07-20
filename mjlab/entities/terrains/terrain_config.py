from dataclasses import dataclass

from mjlab.utils.spec_editor.spec_editor_config import CollisionCfg, GeomCfg
from mjlab.entities.entity_config import EntityCfg


@dataclass
class TerrainCfg(EntityCfg):
  geoms: tuple[GeomCfg, ...] = ()
  collisions: tuple[CollisionCfg, ...] = ()
