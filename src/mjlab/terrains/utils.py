"""Utility functions for terrain generation.

References:
  IsaacLab terrain utilities:
  https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/trimesh/utils.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
  from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg


def find_flat_patches_from_heightfield(
  heights: np.ndarray,
  horizontal_scale: float,
  z_offset: float,
  cfg: FlatPatchSamplingCfg,
  rng: np.random.Generator,
) -> np.ndarray:
  """Find flat patches on a heightfield surface using morphological filtering.

  Slides a circular footprint over every pixel and keeps those where the
  height variation is within tolerance. Randomly samples from the valid set.

  Args:
    heights: 2D array of physical heights (meters), shape (num_rows, num_cols).
    horizontal_scale: Physical size of each pixel in meters.
    z_offset: Vertical offset of the heightfield origin.
    cfg: Flat patch sampling configuration.
    rng: Random number generator.

  Returns:
    Array of shape (num_patches, 3) with (x, y, z) positions in sub-terrain
    local frame.
  """
  # Optionally upsample to a finer grid for higher-precision boundary detection.
  if cfg.grid_resolution is not None and cfg.grid_resolution < horizontal_scale:
    zoom_factor = horizontal_scale / cfg.grid_resolution
    heights = np.asarray(ndimage.zoom(heights, zoom_factor, order=1))
    horizontal_scale = cfg.grid_resolution

  num_rows, num_cols = heights.shape

  # Build circular footprint.
  radius_pixels = int(np.ceil(cfg.patch_radius / horizontal_scale))
  y_grid, x_grid = np.ogrid[
    -radius_pixels : radius_pixels + 1, -radius_pixels : radius_pixels + 1
  ]
  footprint = (x_grid**2 + y_grid**2) <= radius_pixels**2

  # Morphological max/min filter to find height variation within footprint.
  # Use constant padding so edge pixels where the footprint extends outside
  # the data are not incorrectly marked flat (default 'reflect' hides edges).
  max_h = ndimage.maximum_filter(
    heights, footprint=footprint, mode="constant", cval=-np.inf
  )
  min_h = ndimage.minimum_filter(
    heights, footprint=footprint, mode="constant", cval=np.inf
  )
  valid_mask = (max_h - min_h) <= cfg.max_height_diff

  # Exclude pixels whose footprint would extend outside the array. This
  # ensures the full patch circle lies within the heightfield bounds.
  valid_mask[:radius_pixels, :] = False
  valid_mask[-radius_pixels:, :] = False
  valid_mask[:, :radius_pixels] = False
  valid_mask[:, -radius_pixels:] = False

  # Apply spatial range constraints.
  # MuJoCo hfield convention: columns map to the x-axis, rows map to the
  # y-axis (see engine_ray.c vertex: {dx*c - size[0], dy*r - size[1], ...}).
  x_coords = np.arange(num_cols) * horizontal_scale
  y_coords = np.arange(num_rows) * horizontal_scale

  x_valid = (x_coords >= cfg.x_range[0]) & (x_coords <= cfg.x_range[1])
  y_valid = (y_coords >= cfg.y_range[0]) & (y_coords <= cfg.y_range[1])
  valid_mask &= y_valid[:, None] & x_valid[None, :]

  # Apply z range constraint.
  z_values = heights + z_offset
  z_valid = (z_values >= cfg.z_range[0]) & (z_values <= cfg.z_range[1])
  valid_mask &= z_valid

  valid_indices = np.argwhere(valid_mask)

  if len(valid_indices) == 0:
    # Fallback: return sub-terrain center repeated.
    center_x = num_cols * horizontal_scale / 2.0
    center_y = num_rows * horizontal_scale / 2.0
    center_row = min(num_rows // 2, num_rows - 1)
    center_col = min(num_cols // 2, num_cols - 1)
    center_z = heights[center_row, center_col] + z_offset
    return np.tile([center_x, center_y, center_z], (cfg.num_patches, 1))

  replace = len(valid_indices) < cfg.num_patches
  chosen = rng.choice(len(valid_indices), size=cfg.num_patches, replace=replace)
  selected = valid_indices[chosen]

  x = selected[:, 1] * horizontal_scale
  y = selected[:, 0] * horizontal_scale
  z = heights[selected[:, 0], selected[:, 1]] + z_offset

  return np.stack([x, y, z], axis=-1)


def make_plane(
  body: mujoco.MjsBody,
  size: tuple[float, float],
  height: float,
  center_zero: bool = True,
  plane_thickness: float = 1.0,
):
  """Create finite plane using box geometry.

  Uses box instead of MuJoCo plane to avoid infinite extent in terrain grids.
  Thickness prevents penetration issues.
  """
  if center_zero:
    pos = (0, 0, height - plane_thickness / 2.0)
  else:
    pos = (size[0] / 2.0, size[1] / 2.0, height - plane_thickness / 2.0)

  box = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(size[0] / 2.0, size[1] / 2.0, plane_thickness / 2.0),
    pos=pos,
  )
  return [box]


def make_border(
  body: mujoco.MjsBody,
  size: tuple[float, float],
  inner_size: tuple[float, float],
  height: float,
  position: tuple[float, float, float],
):
  """Create rectangular border using four box geometries.

  Returns top, bottom, left, right boxes forming a hollow rectangle.
  """
  boxes = []

  thickness_x = (size[0] - inner_size[0]) / 2.0
  thickness_y = (size[1] - inner_size[1]) / 2.0

  box_dims = (size[0], thickness_y, height)

  # Top.
  box_pos = (
    position[0],
    position[1] + inner_size[1] / 2.0 + thickness_y / 2.0,
    position[2],
  )
  box = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
    pos=box_pos,
  )
  boxes.append(box)

  # Bottom.
  box_pos = (
    position[0],
    position[1] - inner_size[1] / 2.0 - thickness_y / 2.0,
    position[2],
  )
  box = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
    pos=box_pos,
  )
  boxes.append(box)

  box_dims = (thickness_x, inner_size[1], height)

  # Left.
  box_pos = (
    position[0] - inner_size[0] / 2.0 - thickness_x / 2.0,
    position[1],
    position[2],
  )
  box = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
    pos=box_pos,
  )
  boxes.append(box)

  # Right.
  box_pos = (
    position[0] + inner_size[0] / 2.0 + thickness_x / 2.0,
    position[1],
    position[2],
  )
  box = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
    pos=box_pos,
  )
  boxes.append(box)

  return boxes
