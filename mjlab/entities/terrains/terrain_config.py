from dataclasses import dataclass, field

from mjlab.utils.spec_editor.spec_editor_config import CollisionCfg, GeomCfg
from mjlab.entities.entity_config import EntityCfg


@dataclass
class TerrainCfg(EntityCfg):
  geoms: list[GeomCfg] = field(default_factory=list)
  collisions: list[CollisionCfg] = field(default_factory=list)
