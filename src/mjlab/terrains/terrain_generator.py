# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from mjlab.terrains.utils import make_border
from mjlab.utils.color import RGBA

_DARK_GRAY = (0.2, 0.2, 0.2, 1.0)


@dataclass
class TerrainGeometry:
  geom: mujoco.MjsGeom | None = None
  hfield: mujoco.MjsHField | None = None
  color: tuple[float, float, float, float] | None = None


@dataclass
class TerrainOutput:
  origin: np.ndarray
  geometries: list[TerrainGeometry]


@dataclass
class SubTerrainCfg(abc.ABC):
  proportion: float = 1.0
  size: tuple[float, float] = (10.0, 10.0)

  @abc.abstractmethod
  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    """Generate terrain geometry.

    Returns:
      TerrainOutput containing spawn origin and list of geometries.
    """
    raise NotImplementedError


@dataclass(kw_only=True)
class TerrainGeneratorCfg:
  seed: int | None = None
  curriculum: bool = False
  size: tuple[float, float]
  border_width: float = 0.0
  border_height: float = 1.0
  num_rows: int = 1
  num_cols: int = 1
  color_scheme: Literal["height", "random", "none"] = "height"
  sub_terrains: dict[str, SubTerrainCfg]
  difficulty_range: tuple[float, float] = (0.0, 1.0)
  add_lights: bool = False


class TerrainGenerator:
  """Generates procedural terrain grids with configurable difficulty.

  Creates a grid of terrain patches where each patch can be a different
  terrain type. Supports two modes: random (patches get random difficulty)
  or curriculum (difficulty increases along rows for progressive training).

  Terrain types are weighted by proportion and their geometry is generated
  based on a difficulty value in the configured range. The grid is centered
  at the world origin. A border can be added around the entire grid along with
  optional overhead lighting.
  """

  def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu") -> None:
    if len(cfg.sub_terrains) == 0:
      raise ValueError("At least one sub_terrain must be specified.")

    self.cfg = cfg
    self.device = device

    for sub_cfg in self.cfg.sub_terrains.values():
      sub_cfg.size = self.cfg.size

    if self.cfg.seed is not None:
      seed = self.cfg.seed
    else:
      seed = np.random.randint(0, 10000)
    self.np_rng = np.random.default_rng(seed)

    self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

  def compile(self, spec: mujoco.MjSpec) -> None:
    body = spec.worldbody.add_body(name="terrain")

    if self.cfg.curriculum:
      tic = time.perf_counter()
      self._generate_curriculum_terrains(spec)
      toc = time.perf_counter()
      print(f"Curriculum terrain generation took {toc - tic:.4f} seconds.")

    else:
      tic = time.perf_counter()
      self._generate_random_terrains(spec)
      toc = time.perf_counter()
      print(f"Terrain generation took {toc - tic:.4f} seconds.")

    self._add_terrain_border(spec)
    self._add_grid_lights(spec)

    counter = 0
    for geom in body.geoms:
      geom.name = f"terrain_{counter}"
      counter += 1

  def _generate_random_terrains(self, spec: mujoco.MjSpec) -> None:
    # Normalize the proportions of the sub-terrains.
    proportions = np.array(
      [sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()]
    )
    proportions /= np.sum(proportions)

    sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

    # Randomly sample and place sub-terrains in the grid.
    for index in range(self.cfg.num_rows * self.cfg.num_cols):
      sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
      sub_row = int(sub_row)
      sub_col = int(sub_col)

      # Randomly select a sub-terrain type and difficulty.
      sub_index = self.np_rng.choice(len(proportions), p=proportions)
      difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)

      # Calculate the world position for this sub-terrain.
      world_position = self._get_sub_terrain_position(sub_row, sub_col)

      # Create the terrain mesh and get the spawn origin in world coordinates.
      spawn_origin = self._create_terrain_geom(
        spec,
        world_position,
        difficulty,
        sub_terrains_cfgs[sub_index],
      )

      # Store the spawn origin for this terrain.
      self.terrain_origins[sub_row, sub_col] = spawn_origin

  def _generate_curriculum_terrains(self, spec: mujoco.MjSpec) -> None:
    # Normalize the proportions of the sub-terrains.
    proportions = np.array(
      [sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()]
    )
    proportions /= np.sum(proportions)

    sub_indices = []
    for index in range(self.cfg.num_cols):
      sub_index = np.min(
        np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0]
      )
      sub_indices.append(sub_index)
    sub_indices = np.array(sub_indices, dtype=np.int32)

    sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

    for sub_col in range(self.cfg.num_cols):
      for sub_row in range(self.cfg.num_rows):
        lower, upper = self.cfg.difficulty_range
        difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
        difficulty = lower + (upper - lower) * difficulty
        world_position = self._get_sub_terrain_position(sub_row, sub_col)
        spawn_origin = self._create_terrain_geom(
          spec, world_position, difficulty, sub_terrains_cfgs[sub_indices[sub_col]]
        )
        self.terrain_origins[sub_row, sub_col] = spawn_origin

  def _get_sub_terrain_position(self, row: int, col: int) -> np.ndarray:
    """Get the world position for a sub-terrain at the given grid indices.

    This returns the position of the sub-terrain's corner (not center).
    The entire grid is centered at the world origin.
    """
    # Calculate position relative to grid corner.
    rel_x = row * self.cfg.size[0]
    rel_y = col * self.cfg.size[1]

    # Offset to center the entire grid at world origin.
    grid_offset_x = -self.cfg.num_rows * self.cfg.size[0] * 0.5
    grid_offset_y = -self.cfg.num_cols * self.cfg.size[1] * 0.5

    return np.array([grid_offset_x + rel_x, grid_offset_y + rel_y, 0.0])

  def _create_terrain_geom(
    self,
    spec: mujoco.MjSpec,
    world_position: np.ndarray,
    difficulty: float,
    cfg: SubTerrainCfg,
  ) -> np.ndarray:
    """Create a terrain geometry at the specified world position.

    Args:
      spec: MuJoCo spec to add geometry to.
      world_position: World position of the terrain's corner.
      difficulty: Difficulty parameter for terrain generation.
      cfg: Sub-terrain configuration.

    Returns:
      The spawn origin in world coordinates.
    """
    output = cfg.function(difficulty, spec, self.np_rng)
    for terrain_geom in output.geometries:
      if terrain_geom.geom is not None:
        terrain_geom.geom.pos = np.array(terrain_geom.geom.pos) + world_position
        if terrain_geom.geom.material is not None:
          if self.cfg.color_scheme == "height" and terrain_geom.color:
            terrain_geom.geom.rgba[:] = terrain_geom.color
          elif self.cfg.color_scheme == "random":
            terrain_geom.geom.rgba[:3] = self.np_rng.uniform(0.3, 0.8, 3)
            terrain_geom.geom.rgba[3] = 1.0
          elif self.cfg.color_scheme == "none":
            terrain_geom.geom.rgba[:] = (0.5, 0.5, 0.5, 1.0)
    return output.origin + world_position

  def _add_terrain_border(self, spec: mujoco.MjSpec) -> None:
    body = spec.body("terrain")
    border_size = (
      self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
      self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
    )
    inner_size = (
      self.cfg.num_rows * self.cfg.size[0],
      self.cfg.num_cols * self.cfg.size[1],
    )
    # Border should be centered at origin since the terrain grid is centered.
    border_center = (0, 0, -self.cfg.border_height / 2)
    boxes = make_border(
      body,
      border_size,
      inner_size,
      height=abs(self.cfg.border_height),
      position=border_center,
    )
    for box in boxes:
      if self.cfg.color_scheme == "random":
        box.rgba = RGBA.random(self.np_rng, alpha=1.0)
      else:
        box.rgba = _DARK_GRAY

  def _add_grid_lights(self, spec: mujoco.MjSpec) -> None:
    if not self.cfg.add_lights:
      return

    total_width = self.cfg.size[0] * self.cfg.num_rows
    total_height = self.cfg.size[1] * self.cfg.num_cols

    light_height = max(total_width, total_height) * 0.6

    positions = [
      (0, 0),  # Center.
      (-total_width * 0.5, -total_height * 0.5),  # Bottom-left.
      (-total_width * 0.5, total_height * 0.5),  # Top-left.
      (total_width * 0.5, -total_height * 0.5),  # Bottom-right.
      (total_width * 0.5, total_height * 0.5),  # Top-right.
    ]

    for i, (x, y) in enumerate(positions):
      intensity = 0.4 if i == 0 else 0.2

      spec.body("terrain").add_light(
        pos=(x, y, light_height),
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        diffuse=(intensity, intensity, intensity * 0.95),
        specular=(0.1, 0.1, 0.1),
        cutoff=70,
        exponent=2,
      )
