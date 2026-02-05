from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
import torch

from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
from mjlab.utils import spec_config as spec_cfg

_DEFAULT_PLANE_TEXTURE = spec_cfg.TextureCfg(
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

_DEFAULT_PLANE_MATERIAL = spec_cfg.MaterialCfg(
  name="groundplane",
  texuniform=True,
  texrepeat=(4, 4),
  reflectance=0.2,
  texture="groundplane",
)


@dataclass
class TerrainImporterCfg:
  """Configuration for terrain import and environment placement."""

  terrain_type: Literal["generator", "plane"] = "plane"
  """Type of terrain to generate. "generator" uses procedural terrain with
  sub-terrain grid, "plane" creates a flat ground plane."""
  terrain_generator: TerrainGeneratorCfg | None = None
  """Configuration for procedural terrain generation. Required when
  terrain_type is "generator"."""
  env_spacing: float | None = 2.0
  """Distance between environment origins when using grid layout. Required for
  "plane" terrain or when no sub-terrain origins exist."""
  max_init_terrain_level: int | None = None
  """Maximum initial difficulty level (row index) for environment placement in
  curriculum mode. None uses all available rows."""
  num_envs: int = 1
  """Number of parallel environments to create. This will get overridden by the
  scene configuration if specified there."""


class TerrainImporter:
  """Builds a MuJoCo spec with terrain geometry and maps environments to spawn origins.

  The terrain is a grid of sub-terrain patches (num_rows x num_cols), each with
  a spawn origin. When num_envs exceeds the number of patches, environment
  origins are sampled from the sub-terrain origins.

  .. note::
    Environment allocation for procedural terrain: Columns (terrain types) are
    evenly distributed across environments, but rows (difficulty levels) are
    randomly sampled. This means multiple environments can spawn on the same
    (row, col) patch, leaving others unoccupied, even when num_envs > num_patches.

  See FAQ: "How does env_origins determine robot layout?"
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
      self._flat_patches: dict[str, torch.Tensor] = {
        name: torch.from_numpy(arr).to(device=device, dtype=torch.float)
        for name, arr in terrain_generator.flat_patches.items()
      }
      self._flat_patch_radii: dict[str, float] = dict(
        terrain_generator.flat_patch_radii
      )
    elif self.cfg.terrain_type == "plane":
      self.import_ground_plane("terrain")
      self.configure_env_origins()
      self._flat_patches: dict[str, torch.Tensor] = {}
      self._flat_patch_radii: dict[str, float] = {}
    else:
      raise ValueError(f"Unknown terrain type: {self.cfg.terrain_type}")

    self._add_env_origin_sites()
    self._add_terrain_origin_sites()
    self._add_flat_patch_sites()

  def _add_env_origin_sites(self) -> None:
    """Add transparent sphere sites at each environment origin for visualization."""
    if self.env_origins is None:
      return

    origin_site_radius: float = 0.3
    origin_site_color: tuple[float, float, float, float] = (0.2, 0.6, 0.2, 0.3)

    # Convert torch tensor to numpy if needed
    if isinstance(self.env_origins, torch.Tensor):
      env_origins_np = self.env_origins.cpu().numpy()
    else:
      env_origins_np = self.env_origins

    for env_id, origin in enumerate(env_origins_np):
      self._spec.worldbody.add_site(
        name=f"env_origin_{env_id}",
        pos=origin,
        size=(origin_site_radius,) * 3,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=origin_site_color,
        group=4,
      )

  def _add_terrain_origin_sites(self) -> None:
    """Add transparent sphere sites at each terrain origin for visualization."""
    if self.terrain_origins is None:
      return

    # Convert torch tensor to numpy if needed
    if isinstance(self.terrain_origins, torch.Tensor):
      terrain_origins_np = self.terrain_origins.cpu().numpy()
    else:
      terrain_origins_np = self.terrain_origins

    terrain_origin_site_radius: float = 0.5
    terrain_origin_site_color: tuple[float, float, float, float] = (0.2, 0.2, 0.6, 0.3)

    # Iterate through the 2D grid of terrain origins
    num_rows, num_cols = terrain_origins_np.shape[:2]
    for row in range(num_rows):
      for col in range(num_cols):
        origin = terrain_origins_np[row, col]
        self._spec.worldbody.add_site(
          name=f"terrain_origin_{row}_{col}",
          pos=origin,
          size=(terrain_origin_site_radius,) * 3,
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          rgba=terrain_origin_site_color,
          group=5,
        )

  def _add_flat_patch_sites(self) -> None:
    """Add transparent box sites at each flat patch location for visualization."""
    if not self._flat_patches:
      return
    site_thickness = 0.02
    site_color = (0.9, 0.6, 0.1, 0.1)
    for name, patches_tensor in self._flat_patches.items():
      radius = self._flat_patch_radii.get(name, 0.5)
      patches_np = patches_tensor.cpu().numpy()
      num_rows, num_cols, num_patches, _ = patches_np.shape
      for row in range(num_rows):
        for col in range(num_cols):
          for p in range(num_patches):
            pos = patches_np[row, col, p]
            self._spec.worldbody.add_site(
              name=f"flat_patch_{name}_{row}_{col}_{p}",
              pos=pos,
              size=(radius, radius, site_thickness),
              type=mujoco.mjtGeom.mjGEOM_BOX,
              rgba=site_color,
              group=3,
            )

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def flat_patches(self) -> dict[str, torch.Tensor]:
    return self._flat_patches

  def import_ground_plane(self, name: str) -> None:
    _DEFAULT_PLANE_TEXTURE.edit_spec(self._spec)
    _DEFAULT_PLANE_MATERIAL.edit_spec(self._spec)
    self._spec.worldbody.add_body(name=name).add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=(0, 0, 0.01),
      material=_DEFAULT_PLANE_MATERIAL.name,
    )
    spec_cfg.LightCfg(pos=(0, 0, 1.5), type="directional").edit_spec(self._spec)

  def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
    """Configure the origins of the environments based on the added terrain."""
    if origins is not None:
      if isinstance(origins, np.ndarray):
        origins = torch.from_numpy(origins)
      else:
        assert isinstance(origins, torch.Tensor)
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

  def randomize_env_origins(self, env_ids: torch.Tensor) -> None:
    """Randomize the environment origins to random sub-terrains.

    This randomizes both the terrain level (row) and terrain type (column),
    useful for play/evaluation mode where you want to test on varied terrains.
    """
    if self.terrain_origins is None:
      return
    assert self.env_origins is not None
    num_rows, num_cols = self.terrain_origins.shape[:2]
    num_envs = len(env_ids)
    self.terrain_levels[env_ids] = torch.randint(
      0, num_rows, (num_envs,), device=self.device
    )
    self.terrain_types[env_ids] = torch.randint(
      0, num_cols, (num_envs,), device=self.device
    )
    self.env_origins[env_ids] = self.terrain_origins[
      self.terrain_levels[env_ids], self.terrain_types[env_ids]
    ]

  def _compute_env_origins_curriculum(
    self, num_envs: int, origins: torch.Tensor
  ) -> torch.Tensor:
    """Compute the origins of the environments defined by the sub-terrains origins.

    Allocation strategy:
      - Columns (terrain_types): Evenly distributed across environments using
        integer division. Each column gets floor(num_envs / num_cols) or
        ceil(num_envs / num_cols) envs.
      - Rows (terrain_levels): Randomly sampled from [0, max_init_terrain_level].
        Supports curriculum learning where rows represent difficulty levels.

    .. note::
      Multiple environments can be assigned to the same (row, col) patch, leaving
      other patches unoccupied, even when num_envs > num_patches. This is because
      row assignment is random while column assignment is deterministic.

    Example: 5x5 terrain grid (25 patches), 100 environments:
      - Each of 5 columns gets exactly 20 environments
      - Those 20 are randomly distributed across 5 rows
      - Result: Some patches get 0 envs, others might get 5+

    See FAQ: "How does env_origins determine robot layout?"
    """
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
    """Compute the origins of the environments in a grid based on configured spacing.

    Creates an approximately square grid centered at the world origin where:
      - num_rows ≈ ceil(sqrt(num_envs))
      - num_cols = ceil(num_envs / num_rows)

    Examples:
      - 32 envs → 7 rows x 5 cols
      - 64 envs → 8 rows x 8 cols
      - 4096 envs → 64 rows x 64 cols

    See FAQ: "How does env_origins determine robot layout?"
    """
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
