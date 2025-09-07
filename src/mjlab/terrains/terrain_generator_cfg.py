from dataclasses import dataclass
from typing import Literal

from mjlab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from mjlab.terrains.terrain_generator import TerrainGenerator


@dataclass(kw_only=True)
class TerrainGeneratorCfg:
  class_type: type = TerrainGenerator
  seed: int | None = None
  curriculum: bool = False
  size: tuple[float, float]
  border_width: float = 0.0
  border_height: float = 1.0
  num_rows: int = 1
  num_cols: int = 1
  color_scheme: Literal["height", "random", "none"] = "none"
  horizontal_scale: float = 0.1
  vertical_scale: float = 0.005
  slope_threshold: float | None = 0.75
  sub_terrains: dict[str, SubTerrainBaseCfg]
  difficulty_range: tuple[float, float] = (0.0, 1.0)
  use_cache: bool = False
  cache_dir: str = "/tmp/mjlab/terrains"
