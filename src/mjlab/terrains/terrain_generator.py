from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from mjlab.third_party.isaaclab.isaaclab.terrains.trimesh.utils import make_border

if TYPE_CHECKING:
  from mjlab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
  from mjlab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


class TerrainGenerator:
  """Terrain generator to handle different terrain generation functions."""

  terrain_mesh: trimesh.Trimesh
  terrain_meshes: list[trimesh.Trimesh]
  terrain_origins: np.ndarray

  def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu") -> None:
    if len(cfg.sub_terrains) == 0:
      raise ValueError("At least one sub_terrain must be specified.")

    self.cfg = cfg
    self.device = device

    if self.cfg.use_cache and self.cfg.seed is None:
      raise ValueError("Seed must be specified if use_cache is True.")

    if self.cfg.seed is not None:
      seed = self.cfg.seed
    else:
      seed = np.random.randint(0, 10000)
    self.np_rng = np.random.default_rng(seed)

    self.terrain_meshes = list()
    self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

    if self.cfg.curriculum:
      raise NotImplementedError("Curriculum mode is not implemented yet.")
    else:
      self._generate_random_terrains()
    self._add_terrain_border()
    self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

    # Offset the entire terrain and origins so that it is centered.
    transform = np.eye(4)
    transform[:2, -1] = (
      -self.cfg.size[0] * self.cfg.num_rows * 0.5,
      -self.cfg.size[1] * self.cfg.num_cols * 0.5,
    )
    self.terrain_mesh.apply_transform(transform)
    self.terrain_origins += transform[:3, -1]

  # Private methods.

  def _generate_random_terrains(self) -> None:
    # Normalize the proportions of the sub-terrains.
    proportions = np.array(
      [sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()]
    )
    proportions /= np.sum(proportions)

    sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

    # Randomly sample sub-terrains.
    for index in range(self.cfg.num_rows * self.cfg.num_cols):
      sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
      sub_row = int(sub_row)
      sub_col = int(sub_col)
      sub_index = self.np_rng.choice(len(proportions), p=proportions)
      difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
      meshes, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
      for mesh in meshes:
        self._add_sub_terrain(
          mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index]
        )

  def _add_terrain_border(self) -> None:
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
    border_meshes = make_border(
      border_size,
      inner_size,
      height=abs(self.cfg.border_height),
      position=border_center,
    )
    border = trimesh.util.concatenate(border_meshes)
    selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
    border.update_faces(selector)
    self.terrain_meshes.append(border)

  def _add_sub_terrain(
    self,
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    row: int,
    col: int,
    sub_terrain_cfg: SubTerrainBaseCfg,
  ) -> None:
    """Add input sub-terrain to the list of sub-terrains."""
    transform = np.eye(4)
    transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
    mesh.apply_transform(transform)
    self.terrain_meshes.append(mesh)
    self.terrain_origins[row, col] = origin + transform[:3, -1]

  def _get_terrain_mesh(
    self, difficulty: float, cfg: SubTerrainBaseCfg
  ) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    cfg = replace(cfg)
    meshes, origin = cfg.function(difficulty, cfg)
    # mesh = trimesh.util.concatenate(meshes)
    transform = np.eye(4)
    transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
    for mesh in meshes:
      mesh.apply_transform(transform)
    origin += transform[0:3, -1]
    return meshes, origin
