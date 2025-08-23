from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.entities.entity_config import EntityCfg
from mjlab.utils.spec_editor.spec_editor_config import CollisionCfg, GeomCfg


@dataclass
class TerrainCfg(EntityCfg):
  geoms: tuple[GeomCfg, ...] = field(default_factory=tuple)
  collisions: tuple[CollisionCfg, ...] = field(default_factory=tuple)
