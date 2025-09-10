from __future__ import annotations

import abc
import time
from dataclasses import dataclass, replace
from typing import Literal

import mujoco
import numpy as np

from mjlab.terrains.utils import make_border

_DARK_GRAY = (0.2, 0.2, 0.2, 1.0)


@dataclass
class SubTerrainCfg(abc.ABC):
  proportion: float = 1.0
  size: tuple[float, float] = (10.0, 10.0)

  @abc.abstractmethod
  def function(self, difficulty: float, spec: mujoco.MjSpec):
    """Generate terrain geometry.

    Returns:
      origin: The spawn point in local coordinates (relative to terrain corner).
      boxes: List of box geometries to add to the spec.
      colors: List of colors for each box.
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
  color_scheme: Literal["height", "random", "none"] = "none"
  horizontal_scale: float = 0.1
  vertical_scale: float = 0.005
  slope_threshold: float | None = 0.75
  sub_terrains: dict[str, SubTerrainCfg]
  difficulty_range: tuple[float, float] = (0.0, 1.0)
  add_lights: bool = False


class TerrainGenerator:
  """Terrain generator to handle different terrain generation functions."""

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
    if self.cfg.curriculum:
      raise NotImplementedError("Curriculum mode is not implemented yet.")
    else:
      tic = time.perf_counter()
      self._generate_random_terrains(spec)
      toc = time.perf_counter()
      print(f"Terrain generation took {toc - tic:.4f} seconds.")
    # self._add_terrain_border(spec)
    self._add_grid_lights(spec)

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
      spawn_origin = self._create_terrain_mesh(
        spec, world_position, difficulty, sub_terrains_cfgs[sub_index]
      )

      # Store the spawn origin for this terrain.
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

  def _create_terrain_mesh(
    self,
    spec: mujoco.MjSpec,
    world_position: np.ndarray,
    difficulty: float,
    cfg: SubTerrainCfg,
  ) -> np.ndarray:
    """Create a terrain mesh at the specified world position.

    Args:
      spec: MuJoCo spec to add geometry to.
      world_position: World position of the terrain's corner.
      difficulty: Difficulty parameter for terrain generation.
      cfg: Sub-terrain configuration.

    Returns:
      The spawn origin in world coordinates.
    """
    cfg = replace(cfg)

    # Generate terrain geometry (origin is in local coordinates).
    local_origin, boxes, colors = cfg.function(difficulty, spec)

    # Apply transformations to place geometry in world coordinates.
    for box, color in zip(boxes, colors, strict=True):
      box.pos += world_position
      box.rgba = color

    # Convert local origin to world coordinates.
    return local_origin + world_position

  def _add_terrain_border(self, spec: mujoco.MjSpec) -> None:
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
      spec,
      border_size,
      inner_size,
      height=abs(self.cfg.border_height),
      position=border_center,
    )
    for box in boxes:
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

      spec.worldbody.add_light(
        pos=(x, y, light_height),
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        diffuse=(intensity, intensity, intensity * 0.95),
        specular=(0.1, 0.1, 0.1),
        cutoff=70,
        exponent=2,
      )
