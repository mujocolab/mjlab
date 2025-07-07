from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco


@dataclass(frozen=True)
class TextureCfg:
  name: str
  type: str
  builtin: str
  mark: str
  rgb1: tuple[float, float, float]
  rgb2: tuple[float, float, float]
  markrgb: tuple[float, float, float]
  width: int
  height: int


@dataclass(frozen=True)
class MaterialCfg:
  name: str
  texuniform: bool
  texrepeat: tuple[int, int]
  reflectance: float = 0.0
  texture: str | None = None


@dataclass(frozen=True)
class GeomCfg:
  name: str
  body: str
  type: str
  size: tuple[int, ...]
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
  construct_fn: Callable[[mujoco.MjSpec], None] | None = None
