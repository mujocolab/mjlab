from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco


@dataclass
class TextureCfg:
  type: str
  builtin: str
  mark: str
  rgb1: tuple[float, float, float]
  rgb2: tuple[float, float, float]
  markrgb: tuple[float, float, float]
  width: int
  height: int


@dataclass
class MaterialCfg:
  texuniform: bool
  texrepeat: tuple[int, int]
  reflectance: float = 0.0
  texture: str | None = None


@dataclass
class TerrainCfg:
  xml_path: Path | None = None
  asset_fn: Callable[[], dict[str, bytes]] = field(default_factory=lambda: (lambda: {}))
  textures: dict[str, TextureCfg] = field(default_factory=dict)
  materials: dict[str, MaterialCfg] = field(default_factory=dict)
  construct_fn: Callable[[mujoco.MjSpec], None] | None = None
