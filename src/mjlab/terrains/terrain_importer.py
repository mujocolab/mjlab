from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
import trimesh

from mjlab.terrains.terrain_generator import TerrainGenerator

if TYPE_CHECKING:
  from mjlab.terrains.terrain_importer_cfg import TerrainImporterCfg


class TerrainImporter:
  """A class to handle terrain meshes and import them into the simulator."""

  terrain_origins: torch.Tensor | None
  env_origins: torch.Tensor

  def __init__(self, cfg: TerrainImporterCfg) -> None:
    self.cfg = cfg

    self.device = None
    self.env_origins = None

    self._spec = mujoco.MjSpec()

    # auto-import the terrain based on the config
    if self.cfg.terrain_type == "generator":
      if self.cfg.terrain_generator is None:
        raise ValueError(
          "terrain_generator must be specified for terrain_type 'generator'"
        )
      terrain_generator: TerrainGenerator = self.cfg.terrain_generator.class_type(
        cfg=self.cfg.terrain_generator,
        # device=self.device,
      )
      # self.import_mesh("terrain", terrain_generator.terrain_mesh)
      # self.configure_env_origins(terrain_generator.terrain_origins)

      for i, terrain_mesh in enumerate(terrain_generator.terrain_meshes[:-1]):
        self.import_mesh(f"terrain_{i}", terrain_mesh)

    elif self.cfg.terrain_type == "plane":
      raise NotImplementedError("Plane terrain not implemented yet.")
    else:
      raise ValueError(f"Unknown terrain type: {self.cfg.terrain_type}")

  def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
    self._spec.add_mesh(
      name=name,
      uservert=mesh.vertices.flatten(),
      userface=mesh.faces.flatten(),
    )
    self._spec.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=name,
    )

  def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
    # if origins is not None:
    #   if isinstance(origins, np.ndarray):
    #     origins = torch.from_numpy(origins)
    #   self.terrain_origins = origins.to(self.device, dtype=torch.float)
    #   self.env_origins = self._compute_env_origins_curriculum(
    #     self.cfg.num_envs, self.terrain_origins
    #   )
    # else:
    #   self.terrain_origins = None
    #   if self.cfg.env_spacing is None:
    #     raise ValueError(
    #       "Environment spacing must be specified for configuring grid-like origins."
    #     )
    #   self.env_origins = self._compute_env_origins_grid(
    #     self.cfg.num_envs, self.cfg.env_spacing
    #   )
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
    pass

  def _compute_env_origins_curriculum(
    self, num_envs: int, origins: torch.Tensor
  ) -> torch.Tensor:
    raise NotImplementedError("Curriculum-based terrain origins not implemented yet.")

  def _compute_env_origins_grid(
    self, num_envs: int, env_spacing: float
  ) -> torch.Tensor:
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
