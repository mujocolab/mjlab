from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.entity import GeometryIndexing, SceneElement, SceneElementCfg
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
from mjlab.utils import spec_config as spec_cfg

if TYPE_CHECKING:
  import mujoco_warp as mjwarp

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
class TerrainEntityCfg(SceneElementCfg):
  """Configuration for terrain entity.

  TerrainEntityCfg extends SceneElementCfg to enable terrain-specific settings
  while inheriting editor support (textures, materials, lights, cameras,
  collisions).
  """

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
  """Number of parallel environments to create. This will get overriden by the
  scene configuration if specified there."""

  def build(self, device: str) -> TerrainEntity:
    """Build TerrainEntity from this config."""
    return TerrainEntity(self, device)


class TerrainEntity(SceneElement[TerrainEntityCfg]):
  """Scene element representing terrain geometry.

  TerrainEntity is a specialized scene element for static terrain. Unlike
  Entity, it:
  - Has no joints or actuators
  - Is static (no physics simulation on the terrain itself)
  - Manages environment origins for placing other entities
  - Supports curriculum learning through terrain levels

  As a SceneElement, it supports editors (textures, materials, lights, cameras,
  collisions) for randomization.
  """

  def __init__(self, cfg: TerrainEntityCfg, device: str) -> None:
    spec = mujoco.MjSpec()
    super().__init__(cfg, spec)
    self._device = device

    self._terrain_generator: TerrainGenerator | None = None

    # Generate terrain geometry and compute origins in torch
    if self.cfg.terrain_type == "generator":
      if self.cfg.terrain_generator is None:
        raise ValueError(
          "terrain_generator must be specified for terrain_type 'generator'"
        )
      self._terrain_generator = TerrainGenerator(
        self.cfg.terrain_generator, device="cpu"
      )
      self._terrain_generator.compile(self._spec)

      terrain_origins = torch.from_numpy(self._terrain_generator.terrain_origins).to(
        device, dtype=torch.float
      )

      env_origins, terrain_levels, terrain_types = self._compute_curriculum(
        terrain_origins, cfg.num_envs
      )
      self._terrain_origins: torch.Tensor | None = terrain_origins
      self._terrain_levels: torch.Tensor | None = terrain_levels
      self._terrain_types: torch.Tensor | None = terrain_types
      self._max_terrain_level: int | None = terrain_origins.shape[0]

    elif self.cfg.terrain_type == "plane":
      self._import_ground_plane()
      env_origins = self._compute_grid(cfg.num_envs, cfg.env_spacing or 2.0)
      self._terrain_origins = None
      self._terrain_levels = None
      self._terrain_types = None
      self._max_terrain_level = None

    else:
      raise ValueError(f"Unknown terrain type: {self.cfg.terrain_type}")

    self._env_origins = env_origins

    # Add visualization sites (convert torch -> numpy for pos=)
    self._add_env_origin_sites(env_origins.cpu().numpy())
    if self._terrain_origins is not None:
      self._add_terrain_origin_sites(self._terrain_origins.cpu().numpy())

    # Apply editors (textures, materials, lights, cameras, collisions)
    self._apply_spec_editors()

  def _import_ground_plane(self) -> None:
    """Import default ground plane with texture and material."""
    _DEFAULT_PLANE_TEXTURE.edit_spec(self._spec)
    _DEFAULT_PLANE_MATERIAL.edit_spec(self._spec)
    self._spec.worldbody.add_body(name="terrain").add_geom(
      name="terrain",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=(0, 0, 0.01),
      material=_DEFAULT_PLANE_MATERIAL.name,
    )
    spec_cfg.LightCfg(pos=(0, 0, 1.5), type="directional").edit_spec(self._spec)

  # Terrain-specific properties

  @property
  def env_origins(self) -> torch.Tensor:
    """Environment spawn positions. Shape: (num_envs, 3).

    For plane terrain, returns a grid layout based on env_spacing.
    For curriculum terrain (generator), returns positions on sub-terrain grid.
    """
    return self._env_origins

  @property
  def terrain_origins(self) -> torch.Tensor | None:
    """Sub-terrain grid origins. Shape: (num_rows, num_cols, 3) or None."""
    return self._terrain_origins

  @property
  def terrain_levels(self) -> torch.Tensor:
    """Current terrain level (row) for each environment. Shape: (num_envs,)."""
    assert self._terrain_levels is not None
    return self._terrain_levels

  @property
  def terrain_types(self) -> torch.Tensor:
    """Current terrain type (column) for each environment. Shape: (num_envs,)."""
    assert self._terrain_types is not None
    return self._terrain_types

  @property
  def max_terrain_level(self) -> int:
    """Maximum terrain level (number of rows)."""
    assert self._max_terrain_level is not None
    return self._max_terrain_level

  @property
  def terrain_generator(self) -> TerrainGenerator | None:
    """Return the terrain generator if using procedural terrain."""
    return self._terrain_generator

  # Lifecycle methods

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    """Initialize terrain indexing for domain randomization."""
    del (
      model,
      data,
      device,
    )  # Unused - terrain is static, device already set in __init__

    # Compute indexing for domain randomization support
    self.indexing = self._compute_indexing(mj_model, self._device)

  def _compute_indexing(
    self, mj_model: mujoco.MjModel, device: str
  ) -> GeometryIndexing:
    """Compute indexing for terrain bodies/geoms/sites.

    This enables domain randomization of terrain properties (friction, colors, etc.)
    using the same API as entities.
    """
    del mj_model  # Unused.

    bodies = tuple(self.spec.bodies[1:])  # Exclude world body
    geoms = tuple(self.spec.geoms)
    sites = tuple(self.spec.sites)

    return GeometryIndexing(
      bodies=bodies,
      geoms=geoms,
      sites=sites,
      body_ids=torch.tensor([b.id for b in bodies], dtype=torch.int, device=device),
      geom_ids=torch.tensor([g.id for g in geoms], dtype=torch.int, device=device),
      site_ids=torch.tensor([s.id for s in sites], dtype=torch.int, device=device),
    )

  # Curriculum methods

  def update_env_origins(
    self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor
  ) -> None:
    """Update environment origins based on curriculum progress.

    Args:
      env_ids: Indices of environments to update.
      move_up: Boolean tensor indicating which envs should move to harder terrain.
      move_down: Boolean tensor indicating which envs should move to easier terrain.
    """
    if self._terrain_origins is None:
      return
    assert self._terrain_levels is not None
    assert self._terrain_types is not None
    assert self._max_terrain_level is not None

    self._terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    self._terrain_levels[env_ids] = torch.where(
      self._terrain_levels[env_ids] >= self._max_terrain_level,
      torch.randint_like(self._terrain_levels[env_ids], self._max_terrain_level),
      torch.clip(self._terrain_levels[env_ids], 0),
    )
    self._env_origins[env_ids] = self._terrain_origins[
      self._terrain_levels[env_ids], self._terrain_types[env_ids]
    ]

  def randomize_env_origins(self, env_ids: torch.Tensor) -> None:
    """Randomize environment origins to random sub-terrains.

    This randomizes both the terrain level (row) and terrain type (column),
    useful for play/evaluation mode where you want to test on varied terrains.

    Args:
      env_ids: Indices of environments to randomize.
    """
    if self._terrain_origins is None:
      return
    assert self._terrain_levels is not None
    assert self._terrain_types is not None

    num_rows, num_cols = self._terrain_origins.shape[:2]
    num_envs = len(env_ids)
    self._terrain_levels[env_ids] = torch.randint(
      0, num_rows, (num_envs,), device=self._device
    )
    self._terrain_types[env_ids] = torch.randint(
      0, num_cols, (num_envs,), device=self._device
    )
    self._env_origins[env_ids] = self._terrain_origins[
      self._terrain_levels[env_ids], self._terrain_types[env_ids]
    ]

  # Private helpers

  def _compute_curriculum(
    self, origins: torch.Tensor, num_envs: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute environment origins from sub-terrain grid for curriculum."""
    num_rows, num_cols = origins.shape[:2]
    if self.cfg.max_init_terrain_level is None:
      max_init_level = num_rows - 1
    else:
      max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)

    terrain_levels = torch.randint(
      0, max_init_level + 1, (num_envs,), device=self._device, dtype=torch.long
    )
    terrain_types = torch.floor(
      torch.arange(num_envs, device=self._device) / (num_envs / num_cols)
    ).long()

    env_origins = origins[terrain_levels, terrain_types]

    return env_origins, terrain_levels, terrain_types

  def _compute_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
    """Compute environment origins in a grid layout."""

    num_rows = math.ceil(num_envs / int(math.sqrt(num_envs)))
    num_cols = math.ceil(num_envs / num_rows)

    ii = torch.arange(num_rows, device=self._device).repeat_interleave(num_cols)
    jj = torch.arange(num_cols, device=self._device).repeat(num_rows)

    env_origins = torch.zeros((num_envs, 3), dtype=torch.float, device=self._device)
    env_origins[:, 0] = -(ii[:num_envs] - (num_rows - 1) / 2) * env_spacing
    env_origins[:, 1] = (jj[:num_envs] - (num_cols - 1) / 2) * env_spacing

    return env_origins

  def _add_env_origin_sites(self, env_origins: np.ndarray) -> None:
    """Add transparent sphere sites at each environment origin for visualization."""
    origin_site_radius: float = 0.3
    origin_site_color: tuple[float, float, float, float] = (0.2, 0.6, 0.2, 0.3)

    for env_id, origin in enumerate(env_origins):
      self._spec.worldbody.add_site(
        name=f"env_origin_{env_id}",
        pos=origin,
        size=(origin_site_radius,) * 3,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=origin_site_color,
        group=4,
      )

  def _add_terrain_origin_sites(self, terrain_origins: np.ndarray) -> None:
    """Add transparent sphere sites at each terrain origin for visualization."""
    terrain_origin_site_radius: float = 0.5
    terrain_origin_site_color: tuple[float, float, float, float] = (0.2, 0.2, 0.6, 0.3)

    num_rows, num_cols = terrain_origins.shape[:2]
    for row in range(num_rows):
      for col in range(num_cols):
        origin = terrain_origins[row, col]
        self._spec.worldbody.add_site(
          name=f"terrain_origin_{row}_{col}",
          pos=origin,
          size=(terrain_origin_site_radius,) * 3,
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          rgba=terrain_origin_site_color,
          group=5,
        )
