from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import trimesh

from mjlab.terrains.utils import make_border

if TYPE_CHECKING:
  from mjlab.terrains.sub_terrain_cfg import SubTerrainCfg


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

    self.flat_patches = {}
    self.terrain_meshes = list()
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

  # Private methods.

  def _generate_random_terrains(self, spec: mujoco.MjSpec) -> None:
    # Normalize the proportions of the sub-terrains.
    proportions = np.array(
      [sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()]
    )
    proportions /= np.sum(proportions)

    sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

    # Randomly sample sub-terrains.
    for index in range(self.cfg.num_rows * self.cfg.num_cols):
      # rgb = self.np_rng.uniform(0.0, 1.0, size=(3,))
      # rgba = np.concatenate([rgb, [1.0]])

      sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
      sub_row = int(sub_row)
      sub_col = int(sub_col)
      sub_index = self.np_rng.choice(len(proportions), p=proportions)
      difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
      # Position of the sub-terrain center.
      pos = (
        (sub_row + 0.5) * self.cfg.size[0],
        (sub_col + 0.5) * self.cfg.size[1],
        0.0,
      )
      # Offset entire terrain so that it is centered.
      pos2 = (
        -self.cfg.size[0] * self.cfg.num_rows * 0.5,
        -self.cfg.size[1] * self.cfg.num_cols * 0.5,
        0.0,
      )
      self._get_terrain_mesh(spec, pos, pos2, difficulty, sub_terrains_cfgs[sub_index])

  def _add_terrain_border(self, spec: mujoco.MjSpec) -> None:
    border_size = (
      self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
      self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
    )
    inner_size = (
      self.cfg.num_rows * self.cfg.size[0],
      self.cfg.num_cols * self.cfg.size[1],
    )
    border_center = (
      self.cfg.num_rows * self.cfg.size[0] / 2,
      self.cfg.num_cols * self.cfg.size[1] / 2,
      -self.cfg.border_height / 2,
    )
    boxes = make_border(
      spec,
      border_size,
      inner_size,
      height=abs(self.cfg.border_height),
      position=border_center,
    )
    for box in boxes:
      pos = (
        -self.cfg.size[0] * self.cfg.num_rows * 0.5,
        -self.cfg.size[1] * self.cfg.num_cols * 0.5,
        0.0,
      )
      box.pos += np.array(pos)

  def _get_terrain_mesh(
    self,
    spec: mujoco.MjSpec,
    pos: tuple[float, float, float],
    pos2: tuple[float, float, float],
    difficulty: float,
    cfg: SubTerrainCfg,
  ) -> np.ndarray:
    cfg = replace(cfg)
    origin, boxes, colors = cfg.function(difficulty, spec)
    offset = np.array([-cfg.size[0] * 0.5, -cfg.size[1] * 0.5, 0.0])
    for box, color in zip(boxes, colors, strict=True):
      box.pos += offset + np.array(pos) + np.array(pos2)
      box.rgba = color
    origin += offset + np.array(pos) + np.array(pos2)
    return origin

  def _add_sub_terrain(
    self,
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    row: int,
    col: int,
    sub_terrain_cfg: SubTerrainCfg,
  ) -> None:
    """Add input sub-terrain to the list of sub-terrains."""
    transform = np.eye(4)
    transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
    mesh.apply_transform(transform)
    self.terrain_meshes.append(mesh)
    self.terrain_origins[row, col] = origin + transform[:3, -1]
