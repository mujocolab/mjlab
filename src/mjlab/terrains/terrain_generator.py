from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from mjlab.terrains.utils import make_border
from mjlab.utils.color import RGBA

_DARK_GRAY = (0.2, 0.2, 0.2, 1.0)


@dataclass
class FlatPatchSamplingCfg:
  """Configuration for sampling flat patches on a heightfield surface."""

  num_patches: int = 10
  """Number of flat patches to sample per sub-terrain."""
  patch_radius: float = 0.5
  """Radius of the circular footprint used to test flatness, in meters."""
  max_height_diff: float = 0.05
  """Maximum allowed height variation within the patch footprint, in meters."""
  x_range: tuple[float, float] = (-1e6, 1e6)
  """Allowed range of x coordinates for sampled patches, in meters."""
  y_range: tuple[float, float] = (-1e6, 1e6)
  """Allowed range of y coordinates for sampled patches, in meters."""
  z_range: tuple[float, float] = (-1e6, 1e6)
  """Allowed range of z coordinates (world height) for sampled patches, in meters."""
  grid_resolution: float | None = None
  """Resolution of the grid used for flat-patch detection, in meters. When
  ``None`` (default), the terrain's own ``horizontal_scale`` is used. Set to a
  smaller value (e.g. 0.025) for finer boundary precision at the cost of a
  larger intermediate grid."""


@dataclass
class TerrainGeometry:
  geom: mujoco.MjsGeom | None = None
  """MuJoCo geometry spec element, or None."""
  hfield: mujoco.MjsHField | None = None
  """MuJoCo heightfield spec element, or None."""
  color: tuple[float, float, float, float] | None = None
  """RGBA color override for this geometry, or None to use default."""


@dataclass
class TerrainOutput:
  origin: np.ndarray
  """Spawn origin position (x, y, z) in the sub-terrain's local frame."""
  geometries: list[TerrainGeometry]
  """List of geometry elements comprising this terrain."""
  flat_patches: dict[str, np.ndarray] | None = None
  """Named sets of flat patch positions, each an (N, 3) array. None if not configured."""


@dataclass
class SubTerrainCfg(abc.ABC):
  proportion: float = 1.0
  """Terrain type allocation weight (behavior depends on curriculum mode):

  - curriculum=True: Controls column allocation. Normalized proportions determine
    how many columns each terrain type occupies via cumulative distribution.
    Example: proportions [0.5, 0.5] with num_cols=2 gives one terrain per column.

  - curriculum=False: Sampling probability for each patch. Each patch independently
    samples a terrain type weighted by normalized proportions.
  """
  size: tuple[float, float] = (10.0, 10.0)
  """Width and length of the terrain patch, in meters."""
  flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
  """Named flat-patch sampling configurations, or None to disable."""

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
  """Random seed for terrain generation. None uses a random seed."""
  curriculum: bool = False
  """Controls terrain allocation mode:

  - curriculum=True: Each column gets ONE terrain type (deterministic allocation).
    Difficulty increases along rows. Use this to ensure each terrain type occupies
    its own column(s).

  - curriculum=False: Every patch is randomly sampled from all terrain types.
    Proportions control sampling probability. Use this for random variety.

  Example: With 2 terrain types and num_cols=2, curriculum=True gives one terrain
  per column. curriculum=False gives a random mix of both types in all patches.
  """
  size: tuple[float, float]
  """Width and length of each sub-terrain patch, in meters."""
  border_width: float = 0.0
  """Width of the flat border around the entire terrain grid, in meters."""
  border_height: float = 1.0
  """Height of the border wall around the terrain grid, in meters."""
  num_rows: int = 1
  """Number of sub-terrain rows in the grid. Represents difficulty levels in
  curriculum mode. Note: Environments are randomly assigned to rows, so multiple
  envs can share the same patch."""
  num_cols: int = 1
  """Number of sub-terrain columns in the grid. Represents terrain type variants.
  Note: Environments are evenly distributed across columns (not random)."""
  color_scheme: Literal["height", "random", "none"] = "height"
  """Coloring strategy for terrain geometry. "height" colors by elevation,
  "random" assigns random colors, "none" uses uniform gray."""
  sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=dict)
  """Named sub-terrain configurations to populate the grid."""
  difficulty_range: tuple[float, float] = (0.0, 1.0)
  """Min and max difficulty values used when generating sub-terrains."""
  add_lights: bool = False
  """If True, adds a directional light above the terrain grid."""


class TerrainGenerator:
  """Generates procedural terrain grids with configurable difficulty.

  Creates a grid of terrain patches where each patch can be a different
  terrain type. Supports two modes:

  - **Random mode** (curriculum=False): Every patch independently samples a
    terrain type weighted by proportions. Results in random variety across
    all patches.

  - **Curriculum mode** (curriculum=True): Columns are deterministically
    assigned to terrain types based on proportions. All patches in a column
    share the same terrain type, with difficulty increasing along rows.
    Use this to ensure each terrain type occupies specific column(s).

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

    # Pre-allocate flat patch storage by scanning all sub-terrain configs.
    self.flat_patches: dict[str, np.ndarray] = {}
    self.flat_patch_radii: dict[str, float] = {}
    patch_names: dict[str, int] = {}
    for sub_cfg in self.cfg.sub_terrains.values():
      if sub_cfg.flat_patch_sampling is not None:
        for name, patch_cfg in sub_cfg.flat_patch_sampling.items():
          if name in patch_names:
            patch_names[name] = max(patch_names[name], patch_cfg.num_patches)
          else:
            patch_names[name] = patch_cfg.num_patches
          self.flat_patch_radii[name] = max(
            self.flat_patch_radii.get(name, 0.0), patch_cfg.patch_radius
          )
    for name, max_num_patches in patch_names.items():
      self.flat_patches[name] = np.zeros(
        (self.cfg.num_rows, self.cfg.num_cols, max_num_patches, 3)
      )

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
        sub_row,
        sub_col,
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
          spec,
          world_position,
          difficulty,
          sub_terrains_cfgs[sub_indices[sub_col]],
          sub_row,
          sub_col,
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
    sub_row: int,
    sub_col: int,
  ) -> np.ndarray:
    """Create a terrain geometry at the specified world position.

    Args:
      spec: MuJoCo spec to add geometry to.
      world_position: World position of the terrain's corner.
      difficulty: Difficulty parameter for terrain generation.
      cfg: Sub-terrain configuration.
      sub_row: Row index in the terrain grid.
      sub_col: Column index in the terrain grid.

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

    # Collect flat patches into pre-allocated arrays.
    spawn_origin = output.origin + world_position
    for name, arr in self.flat_patches.items():
      if output.flat_patches is not None and name in output.flat_patches:
        patches = output.flat_patches[name]
        arr[sub_row, sub_col, : len(patches)] = patches + world_position
        arr[sub_row, sub_col, len(patches) :] = spawn_origin
      else:
        # Sub-terrain didn't produce patches: fill with spawn origin so that
        # every slot contains a valid position for reset_root_state_from_flat_patches.
        arr[sub_row, sub_col] = spawn_origin

    return spawn_origin

  def _add_terrain_border(self, spec: mujoco.MjSpec) -> None:
    if self.cfg.border_width <= 0.0:
      return
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

    spec.body("terrain").add_light(
      pos=(0, 0, light_height),
      type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
      dir=(0, 0, -1),
    )
