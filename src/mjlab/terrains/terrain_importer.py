from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
import torch

from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
from mjlab.utils.spec_editor import spec_editor, spec_editor_config

_DEFAULT_PLANE_TEXTURE = spec_editor_config.TextureCfg(
  name="groundplane",
  type="2d",
  builtin="checker",
  mark="edge",
  rgb1=(0.2, 0.3, 0.4),
  rgb2=(0.1, 0.2, 0.3),
  markrgb=(0.8, 0.8, 0.8),
  width=300,
  height=300,
)

_DEFAULT_PLANE_MATERIAL = spec_editor_config.MaterialCfg(
  name="groundplane",
  texuniform=True,
  texrepeat=(4, 4),
  reflectance=0.2,
  texture="groundplane",
)


@dataclass
class TerrainImporterCfg:
  terrain_type: Literal["generator", "plane"] = "plane"
  terrain_generator: TerrainGeneratorCfg | None = None
  env_spacing: float | None = 2.0
  max_init_terrain_level: int | None = None
  num_envs: int = 1


class TerrainImporter:
  """A class to handle terrain meshes and import them into the simulator.

  We assume that a terrain mesh comprises of sub-terrains that are arranged in a grid
  with `num_rows` rows and `num_cols` columns. The terrain origins are the positions
  of the sub-terrains where the robot should be spawned.

  Based on the configuration, the terrain importer handles computing the environment
  origins from the sub-terrain origins. In a typical setup, the number of sub-terrains
  `num_rows x num_cols` is smaller than the number of environments `num_envs`. In this
  case, the environment origins are computed by sampling the sub-terrain origins.
  """

  def __init__(self, cfg: TerrainImporterCfg, device: str) -> None:
    self.cfg = cfg
    self.device = device
    self._spec = mujoco.MjSpec()

    # The origins of the environments. Shape is (num_envs, 3).
    self.env_origins = None

    # Origins of the sub-terrains. Shape is (num_rows, num_cols, 3).
    # If terrain origins is not None, the environment origins are computed based on the
    # terrain origins. Otherwise, the origins are computed based on grid spacing.
    self.terrain_origins = None

    if self.cfg.terrain_type == "generator":
      if self.cfg.terrain_generator is None:
        raise ValueError(
          "terrain_generator must be specified for terrain_type 'generator'"
        )
      terrain_generator = TerrainGenerator(self.cfg.terrain_generator, device=device)
      terrain_generator.compile(self._spec)
      self.configure_env_origins(terrain_generator.terrain_origins)
    elif self.cfg.terrain_type == "plane":
      self.import_ground_plane("terrain")
      self.configure_env_origins()
    else:
      raise ValueError(f"Unknown terrain type: {self.cfg.terrain_type}")

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  def import_ground_plane(self, name: str) -> None:
    spec_editor.TextureEditor(_DEFAULT_PLANE_TEXTURE).edit_spec(self._spec)
    spec_editor.MaterialEditor(_DEFAULT_PLANE_MATERIAL).edit_spec(self._spec)
    self._spec.worldbody.add_body(name=name).add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=(0, 0, 0.01),
      material=_DEFAULT_PLANE_MATERIAL.name,
    )

  def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
    """Configure the origins of the environments based on the added terrain."""
    if origins is not None:
      if isinstance(origins, np.ndarray):
        origins = torch.from_numpy(origins)
      self.terrain_origins = origins.to(self.device, dtype=torch.float)
      self.env_origins = self._compute_env_origins_curriculum(
        self.cfg.num_envs, self.terrain_origins
      )
    else:
      self.terrain_origins = None
      if self.cfg.env_spacing is None:
        raise ValueError(
          "Environment spacing must be specified for configuring grid-like origins."
        )
      self.env_origins = self._compute_env_origins_grid(
        self.cfg.num_envs, self.cfg.env_spacing
      )

  def update_env_origins(
    self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor
  ):
    """Update the environment origins based on the terrain levels."""
    if self.terrain_origins is None:
      return
    assert self.env_origins is not None
    self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    self.terrain_levels[env_ids] = torch.where(
      self.terrain_levels[env_ids] >= self.max_terrain_level,
      torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
      torch.clip(self.terrain_levels[env_ids], 0),
    )
    self.env_origins[env_ids] = self.terrain_origins[
      self.terrain_levels[env_ids], self.terrain_types[env_ids]
    ]

  def _compute_env_origins_curriculum(
    self, num_envs: int, origins: torch.Tensor
  ) -> torch.Tensor:
    """Compute the origins of the environments defined by the sub-terrains origins."""
    num_rows, num_cols = origins.shape[:2]
    if self.cfg.max_init_terrain_level is None:
      max_init_level = num_rows - 1
    else:
      max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
    self.max_terrain_level = num_rows
    self.terrain_levels = torch.randint(
      0, max_init_level + 1, (num_envs,), device=self.device
    )
    self.terrain_types = torch.div(
      torch.arange(num_envs, device=self.device),
      (num_envs / num_cols),
      rounding_mode="floor",
    ).to(torch.long)
    env_origins = torch.zeros(num_envs, 3, device=self.device)
    env_origins[:] = origins[self.terrain_levels, self.terrain_types]
    return env_origins

  def _compute_env_origins_grid(
    self, num_envs: int, env_spacing: float
  ) -> torch.Tensor:
    """Compute the origins of the environments in a grid based on configured spacing."""
    env_origins = torch.zeros(num_envs, 3, device=self.device)
    num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
    num_cols = np.ceil(num_envs / num_rows)
    ii, jj = torch.meshgrid(
      torch.arange(num_rows, device=self.device),
      torch.arange(num_cols, device=self.device),
      indexing="ij",
    )
    env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
    env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
    env_origins[:, 2] = 0.0
    return env_origins
