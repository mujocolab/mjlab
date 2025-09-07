from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import trimesh

from mjlab.terrains import mesh_terrains
from mjlab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

T = TypeVar("T", bound=SubTerrainBaseCfg)
FuncType = Callable[[float, T], tuple[list[trimesh.Trimesh], np.ndarray]]


@dataclass(kw_only=True)
class MeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
  function: FuncType = mesh_terrains.pyramid_stairs_terrain
  border_width: float = 0.0
  step_height_range: tuple[float, float]
  step_width: float
  platform_width: float = 1.0
  holes: bool = False


@dataclass(kw_only=True)
class MeshInvertedPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
  function: FuncType = mesh_terrains.inverted_pyramid_stairs_terrain


@dataclass(kw_only=True)
class MeshRandomGridTerrainCfg(SubTerrainBaseCfg):
  function: FuncType = mesh_terrains.random_grid_terrain
  grid_width: float
  grid_height_range: tuple[float, float]
  platform_width: float = 1.0
  holes: bool = False
