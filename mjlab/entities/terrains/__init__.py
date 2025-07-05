"""Terrain entities."""

import enum

from mjlab.entities.terrains.editors import (
  Skybox,
  TerrainEditor,
  RoughTerrain,
  FlatTerrain,
)
from mjlab.entities.terrains.terrain import Terrain, TerrainConfig


class TerrainType(enum.Enum):
  """Terrains we support."""

  FLAT = enum.auto()
  ROUGH = enum.auto()


TERRAIN_PRESETS = {
  TerrainType.FLAT: TerrainConfig(FlatTerrain(), Skybox()),
  TerrainType.ROUGH: TerrainConfig(RoughTerrain()),
}


def get_terrain_config(terrain: TerrainType) -> TerrainConfig:
  return TERRAIN_PRESETS[terrain]


__all__ = (
  # Base classes.
  "Terrain",
  "TerrainConfig",
  # Editors.
  "TerrainEditor",
  "FlatTerrain",
  "RoughTerrain",
  "Skybox",
)
