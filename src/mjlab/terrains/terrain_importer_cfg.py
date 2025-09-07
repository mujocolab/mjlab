from dataclasses import dataclass
from typing import Literal

from mjlab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from mjlab.terrains.terrain_importer import TerrainImporter


@dataclass
class TerrainImporterCfg:
  """Configuration for a terrain manager."""

  class_type: type = TerrainImporter
  collision_group: int = -1
  terrain_type: Literal["generator", "plane"] = "generator"
  terrain_generator: TerrainGeneratorCfg | None = None
  env_spacing: float | None = 2.0
  max_init_terrain_level: int | None = None
  num_envs: int = 1
