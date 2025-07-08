from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco

from mjlab.entities.common.config import TextureCfg, MaterialCfg, CollisionCfg


@dataclass(frozen=True)
class GeomCfg:
  name: str
  type: str
  size: tuple[int, ...]
  body: str = "world"
  rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1)
  material: str | None = None
  group: int = 0


@dataclass
class TerrainCfg:
  xml_path: Path | None = None
  asset_fn: Callable[[], dict[str, bytes]] = field(default_factory=lambda: (lambda: {}))
  textures: tuple[TextureCfg, ...] = ()
  materials: tuple[MaterialCfg, ...] = ()
  geoms: tuple[GeomCfg, ...] = ()
  collisions: tuple[CollisionCfg, ...] = ()
  construct_fn: Callable[[mujoco.MjSpec], None] | None = None
