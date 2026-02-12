"""Terrains composed of primitive geometries.

This module provides terrain generation functionality using primitive geometries,
adapted from the IsaacLab terrain generation system.

References:
  IsaacLab mesh terrain implementation:
  https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_border, make_plane
from mjlab.utils.color import (
  HSV,
  brand_ramp,
  clamp,
  darken_rgba,
  hsv_to_rgb,
  rgb_to_hsv,
)

_MUJOCO_BLUE = (0.20, 0.45, 0.95)
_MUJOCO_RED = (0.90, 0.30, 0.30)
_MUJOCO_GREEN = (0.25, 0.80, 0.45)


def _get_platform_color(
  base_rgb: Tuple[float, float, float],
  desaturation_factor: float = 0.4,
  lightening_factor: float = 0.25,
) -> Tuple[float, float, float, float]:
  hsv = rgb_to_hsv(base_rgb)
  new_s = hsv.s * desaturation_factor
  new_v = clamp(hsv.v + lightening_factor)
  new_hsv = HSV(hsv.h, new_s, new_v)
  r, g, b = hsv_to_rgb(new_hsv)
  return (r, g, b, 1.0)


@dataclass(kw_only=True)
class BoxFlatTerrainCfg(SubTerrainCfg):
  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng  # Unused.
    body = spec.body("terrain")
    origin = (self.size[0] / 2, self.size[1] / 2, 0.0)
    boxes = make_plane(body, self.size, 0.0, center_zero=False)
    box_colors = [(0.5, 0.5, 0.5, 1.0)]
    geometry = TerrainGeometry(geom=boxes[0], color=box_colors[0])
    return TerrainOutput(origin=np.array(origin), geometries=[geometry])


@dataclass(kw_only=True)
class BoxPyramidStairsTerrainCfg(SubTerrainCfg):
  """Configuration for a pyramid stairs terrain."""

  border_width: float = 0.0
  """Width of the flat border frame around the staircase, in meters. Ignored
  when holes is True."""
  step_height_range: tuple[float, float]
  """Min and max step height, in meters. Interpolated by difficulty."""
  step_width: float
  """Depth (run) of each step, in meters."""
  platform_width: float = 1.0
  """Side length of the flat square platform at the top of the staircase, in meters."""
  holes: bool = False
  """If True, steps form a cross pattern with empty gaps in the corners."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng  # Unused.
    boxes = []
    box_colors = []

    body = spec.body("terrain")

    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )

    # Compute number of steps in x and y direction.
    num_steps_x = int(
      (self.size[0] - 2 * self.border_width - self.platform_width)
      / (2 * self.step_width)
    )
    num_steps_y = int(
      (self.size[1] - 2 * self.border_width - self.platform_width)
      / (2 * self.step_width)
    )
    num_steps = max(0, int(min(num_steps_x, num_steps_y)))

    first_step_rgba = brand_ramp(_MUJOCO_BLUE, 0.0)
    border_rgba = darken_rgba(first_step_rgba, 0.85)

    if self.border_width > 0.0 and not self.holes:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -step_height / 2)
      border_inner_size = (
        self.size[0] - 2 * self.border_width,
        self.size[1] - 2 * self.border_width,
      )
      border_boxes = make_border(
        body, self.size, border_inner_size, step_height, border_center
      )
      boxes.extend(border_boxes)
      for _ in range(len(border_boxes)):
        box_colors.append(border_rgba)

    terrain_center = [0.5 * self.size[0], 0.5 * self.size[1], 0.0]
    terrain_size = (
      self.size[0] - 2 * self.border_width,
      self.size[1] - 2 * self.border_width,
    )
    rgba = brand_ramp(_MUJOCO_BLUE, 0.5)
    for k in range(num_steps):
      t = k / max(num_steps - 1, 1)
      rgba = brand_ramp(_MUJOCO_BLUE, t)
      for _ in range(4):
        box_colors.append(rgba)

      if self.holes:
        box_size = (self.platform_width, self.platform_width)
      else:
        box_size = (
          terrain_size[0] - 2 * k * self.step_width,
          terrain_size[1] - 2 * k * self.step_width,
        )
      box_z = terrain_center[2] + k * step_height / 2.0
      box_offset = (k + 0.5) * self.step_width
      box_height = (k + 2) * step_height

      box_dims = (box_size[0], self.step_width, box_height)

      safe_size = (
        np.maximum(1e-6, box_dims[0] / 2.0),
        np.maximum(1e-6, box_dims[1] / 2.0),
        np.maximum(1e-6, box_dims[2] / 2.0),
      )

      # Top.
      box_pos = (
        terrain_center[0],
        terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=box_pos,
      )
      boxes.append(box)

      # Bottom.
      box_pos = (
        terrain_center[0],
        terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=box_pos,
      )
      boxes.append(box)

      if self.holes:
        box_dims = (self.step_width, box_size[1], box_height)
      else:
        box_dims = (
          self.step_width,
          box_size[1] - 2 * self.step_width,
          box_height,
        )
      safe_size = (
        np.maximum(1e-6, box_dims[0] / 2.0),
        np.maximum(1e-6, box_dims[1] / 2.0),
        np.maximum(1e-6, box_dims[2] / 2.0),
      )

      # Right.
      box_pos = (
        terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
        terrain_center[1],
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=box_pos,
      )
      boxes.append(box)

      # Left.
      box_pos = (
        terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
        terrain_center[1],
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=box_pos,
      )
      boxes.append(box)

    # Generate final box for the middle of the terrain.
    box_dims = (
      terrain_size[0] - 2 * num_steps * self.step_width,
      terrain_size[1] - 2 * num_steps * self.step_width,
      (num_steps + 2) * step_height,
    )
    box_pos = (
      terrain_center[0],
      terrain_center[1],
      terrain_center[2] + num_steps * step_height / 2,
    )
    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(
        np.maximum(1e-6, box_dims[0] / 2.0),
        np.maximum(1e-6, box_dims[1] / 2.0),
        np.maximum(1e-6, box_dims[2] / 2.0),
      ),
      pos=box_pos,
    )
    boxes.append(box)
    origin = np.array(
      [terrain_center[0], terrain_center[1], (num_steps + 1) * step_height]
    )
    box_colors.append(rgba)

    geometries = [
      TerrainGeometry(geom=box, color=color)
      for box, color in zip(boxes, box_colors, strict=True)
    ]
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxInvertedPyramidStairsTerrainCfg(BoxPyramidStairsTerrainCfg):
  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng  # Unused.
    boxes = []
    box_colors = []

    body = spec.body("terrain")

    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )

    # Compute number of steps in x and y direction.
    num_steps_x = int(
      (self.size[0] - 2 * self.border_width - self.platform_width)
      / (2 * self.step_width)
    )
    num_steps_y = int(
      (self.size[1] - 2 * self.border_width - self.platform_width)
      / (2 * self.step_width)
    )
    num_steps = max(0, int(min(num_steps_x, num_steps_y)))
    total_height = (num_steps + 1) * step_height

    first_step_rgba = brand_ramp(_MUJOCO_RED, 0.0)
    border_rgba = darken_rgba(first_step_rgba, 0.85)

    if self.border_width > 0.0 and not self.holes:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -0.5 * step_height)
      border_inner_size = (
        self.size[0] - 2 * self.border_width,
        self.size[1] - 2 * self.border_width,
      )
      border_boxes = make_border(
        body, self.size, border_inner_size, step_height, border_center
      )
      boxes.extend(border_boxes)
      for _ in range(len(border_boxes)):
        box_colors.append(border_rgba)

    terrain_center = [0.5 * self.size[0], 0.5 * self.size[1], 0.0]
    terrain_size = (
      self.size[0] - 2 * self.border_width,
      self.size[1] - 2 * self.border_width,
    )

    rgba = brand_ramp(_MUJOCO_RED, 0.5)
    for k in range(num_steps):
      t = k / max(num_steps - 1, 1)
      rgba = brand_ramp(_MUJOCO_RED, t)
      for _ in range(4):
        box_colors.append(rgba)

      if self.holes:
        box_size = (self.platform_width, self.platform_width)
      else:
        box_size = (
          terrain_size[0] - 2 * k * self.step_width,
          terrain_size[1] - 2 * k * self.step_width,
        )

      box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
      box_offset = (k + 0.5) * self.step_width
      box_height = total_height - (k + 1) * step_height

      box_dims = (box_size[0], self.step_width, box_height)
      safe_size = (
        np.maximum(1e-6, box_dims[0] / 2.0),
        np.maximum(1e-6, box_dims[1] / 2.0),
        np.maximum(1e-6, box_dims[2] / 2.0),
      )

      # Top.
      box_pos = (
        terrain_center[0],
        terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=box_pos,
      )
      boxes.append(box)

      # Bottom.
      box_pos = (
        terrain_center[0],
        terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=box_pos,
      )
      boxes.append(box)

      if self.holes:
        box_dims = (self.step_width, box_size[1], box_height)
      else:
        box_dims = (
          self.step_width,
          box_size[1] - 2 * self.step_width,
          box_height,
        )
      safe_size = (
        np.maximum(1e-6, box_dims[0] / 2.0),
        np.maximum(1e-6, box_dims[1] / 2.0),
        np.maximum(1e-6, box_dims[2] / 2.0),
      )

      # Right.
      box_pos = (
        terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
        terrain_center[1],
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=box_pos,
      )
      boxes.append(box)

      # Left.
      box_pos = (
        terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
        terrain_center[1],
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=safe_size,
        pos=box_pos,
      )
      boxes.append(box)

    # Generate final box for the middle of the terrain.
    box_dims = (
      terrain_size[0] - 2 * num_steps * self.step_width,
      terrain_size[1] - 2 * num_steps * self.step_width,
      step_height,
    )
    box_pos = (
      terrain_center[0],
      terrain_center[1],
      terrain_center[2] - total_height - step_height / 2,
    )
    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(
        np.maximum(1e-6, box_dims[0] / 2.0),
        np.maximum(1e-6, box_dims[1] / 2.0),
        np.maximum(1e-6, box_dims[2] / 2.0),
      ),
      pos=box_pos,
    )
    boxes.append(box)
    origin = np.array(
      [terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height]
    )
    box_colors.append(rgba)

    geometries = [
      TerrainGeometry(geom=box, color=color)
      for box, color in zip(boxes, box_colors, strict=True)
    ]
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxRandomGridTerrainCfg(SubTerrainCfg):
  grid_width: float
  """Side length of each square grid cell, in meters."""
  grid_height_range: tuple[float, float]
  """Min and max grid cell height bound, in meters. Interpolated by difficulty.
  At a given difficulty, cell heights are sampled uniformly from
  [-bound, +bound]."""
  platform_width: float = 1.0
  """Side length of the flat square platform at the grid center, in meters."""
  holes: bool = False
  """If True, only the cross-shaped region around the center platform has grid cells."""
  merge_similar_heights: bool = False
  """If True, adjacent cells with similar heights are merged into larger boxes
  to reduce geom count."""
  height_merge_threshold: float = 0.05
  """Maximum height difference between cells that can be merged, in meters."""
  max_merge_distance: int = 3
  """Maximum number of grid cells that can be merged in each direction."""
  border_width: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    if self.size[0] != self.size[1]:
      raise ValueError(f"The terrain must be square. Received size: {self.size}.")

    grid_height = self.grid_height_range[0] + difficulty * (
      self.grid_height_range[1] - self.grid_height_range[0]
    )

    body = spec.body("terrain")

    boxes_list = []
    box_colors = []

    num_boxes_x = int((self.size[0] - 2 * self.border_width) / self.grid_width)
    num_boxes_y = int((self.size[1] - 2 * self.border_width) / self.grid_width)

    terrain_height = 1.0
    border_width = self.size[0] - min(num_boxes_x, num_boxes_y) * self.grid_width

    if border_width <= 0:
      raise RuntimeError(
        "Border width must be greater than 0! Adjust the parameter 'self.grid_width'."
      )

    border_thickness = border_width / 2
    border_center_z = -terrain_height / 2

    half_size = self.size[0] / 2
    half_border = border_thickness / 2
    half_terrain = terrain_height / 2

    first_step_rgba = brand_ramp(_MUJOCO_GREEN, 0.0)
    border_rgba = darken_rgba(first_step_rgba, 0.85)

    border_specs = [
      (
        (half_size, half_border, half_terrain),
        (half_size, self.size[1] - half_border, border_center_z),
      ),
      (
        (half_size, half_border, half_terrain),
        (half_size, half_border, border_center_z),
      ),
      (
        (half_border, (self.size[1] - 2 * border_thickness) / 2, half_terrain),
        (half_border, half_size, border_center_z),
      ),
      (
        (half_border, (self.size[1] - 2 * border_thickness) / 2, half_terrain),
        (self.size[0] - half_border, half_size, border_center_z),
      ),
    ]

    for size, pos in border_specs:
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
      )
      boxes_list.append(box)
      box_colors.append(border_rgba)

    height_map = rng.uniform(-grid_height, grid_height, (num_boxes_x, num_boxes_y))

    if self.merge_similar_heights and not self.holes:
      box_list_, box_color_ = self._create_merged_boxes(
        body,
        height_map,
        num_boxes_x,
        num_boxes_y,
        grid_height,
        terrain_height,
        border_width,
      )
      boxes_list.extend(box_list_)
      box_colors.extend(box_color_)
    else:
      box_list_, box_color_ = self._create_individual_boxes(
        body,
        height_map,
        num_boxes_x,
        num_boxes_y,
        grid_height,
        terrain_height,
        border_width,
      )
      boxes_list.extend(box_list_)
      box_colors.extend(box_color_)

    # Platform
    platform_height = terrain_height + grid_height
    platform_center_z = -terrain_height / 2 + grid_height / 2
    half_platform = self.platform_width / 2

    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(half_platform, half_platform, platform_height / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, platform_center_z),
    )
    boxes_list.append(box)
    platform_rgba = _get_platform_color(_MUJOCO_GREEN)
    box_colors.append(platform_rgba)

    origin = np.array([self.size[0] / 2, self.size[1] / 2, grid_height])

    geometries = [
      TerrainGeometry(geom=box, color=color)
      for box, color in zip(boxes_list, box_colors, strict=True)
    ]
    return TerrainOutput(origin=origin, geometries=geometries)

  def _create_merged_boxes(
    self,
    body,
    height_map,
    num_boxes_x,
    num_boxes_y,
    grid_height,
    terrain_height,
    border_width,
  ):
    """Create merged boxes for similar heights to reduce geom count."""
    boxes = []
    box_colors = []
    visited = np.zeros((num_boxes_x, num_boxes_y), dtype=bool)

    half_border_width = border_width / 2
    neg_half_terrain = -terrain_height / 2

    # Quantize heights to create more merging opportunities
    quantized_heights = (
      np.round(height_map / self.height_merge_threshold) * self.height_merge_threshold
    )

    for i in range(num_boxes_x):
      for j in range(num_boxes_y):
        if visited[i, j]:
          continue

        # Find rectangular region with similar height
        height = quantized_heights[i, j]

        normalized_height = (height + grid_height) / (2 * grid_height)
        t = float(np.clip(normalized_height, 0.0, 1.0))
        rgba = brand_ramp(_MUJOCO_GREEN, t)

        # Greedy expansion in x and y directions
        max_x = i + 1
        max_y = j + 1

        # Try to expand in x direction first
        while max_x < min(i + self.max_merge_distance, num_boxes_x):
          if not visited[max_x, j] and abs(quantized_heights[max_x, j] - height) < 1e-6:
            max_x += 1
          else:
            break

        # Then expand in y direction for the found x range
        can_expand_y = True
        while max_y < min(j + self.max_merge_distance, num_boxes_y) and can_expand_y:
          for x in range(i, max_x):
            if visited[x, max_y] or abs(quantized_heights[x, max_y] - height) > 1e-6:
              can_expand_y = False
              break
          if can_expand_y:
            max_y += 1

        # Mark region as visited
        visited[i:max_x, j:max_y] = True

        # Create merged box
        width_x = (max_x - i) * self.grid_width
        width_y = (max_y - j) * self.grid_width

        box_center_x = half_border_width + (i + (max_x - i) / 2) * self.grid_width
        box_center_y = half_border_width + (j + (max_y - j) / 2) * self.grid_width

        box_height = terrain_height + height
        box_center_z = neg_half_terrain + height / 2

        box = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(width_x / 2, width_y / 2, box_height / 2),
          pos=(box_center_x, box_center_y, box_center_z),
        )
        boxes.append(box)
        box_colors.append(rgba)

    return boxes, box_colors

  def _create_individual_boxes(
    self,
    body,
    height_map,
    num_boxes_x,
    num_boxes_y,
    grid_height,
    terrain_height,
    border_width,
  ):
    """Original approach with individual boxes."""
    boxes = []
    box_colors = []
    half_grid = self.grid_width / 2
    half_border_width = border_width / 2
    neg_half_terrain = -terrain_height / 2

    if self.holes:
      platform_half = self.platform_width / 2
      terrain_center = self.size[0] / 2
      platform_min = terrain_center - platform_half
      platform_max = terrain_center + platform_half
    else:
      platform_min = None
      platform_max = None

    for i in range(num_boxes_x):
      box_center_x = half_border_width + (i + 0.5) * self.grid_width

      if self.holes and not (platform_min <= box_center_x <= platform_max):
        in_y_strip = False
      else:
        in_y_strip = True

      for j in range(num_boxes_y):
        box_center_y = half_border_width + (j + 0.5) * self.grid_width

        if self.holes:
          in_x_strip = platform_min <= box_center_y <= platform_max
          if not (in_x_strip or in_y_strip):
            continue

        height_noise = height_map[i, j]
        box_height = terrain_height + height_noise
        box_center_z = neg_half_terrain + height_noise / 2

        normalized_height = (height_noise + grid_height) / (2 * grid_height)
        t = float(np.clip(normalized_height, 0.0, 1.0))
        rgba = brand_ramp(_MUJOCO_GREEN, t)
        box_colors.append(rgba)

        box = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(half_grid, half_grid, box_height / 2),
          pos=(box_center_x, box_center_y, box_center_z),
        )
        boxes.append(box)

    return boxes, box_colors


@dataclass(kw_only=True)
class BoxRandomSpreadTerrainCfg(SubTerrainCfg):
  num_boxes: int = 60
  box_width_range: tuple[float, float] = (0.3, 1.0)
  box_length_range: tuple[float, float] = (0.3, 1.0)
  box_height_range: tuple[float, float] = (0.05, 1.0)
  box_yaw_range: tuple[float, float] = (0, 360)
  add_floor: bool = True
  platform_width: float = 1.0
  border_width: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    # Scale number of boxes by difficulty.
    num_boxes = int(self.num_boxes * (0.5 + 0.5 * difficulty))

    terrain_height = 1.0
    border_rgba = darken_rgba(brand_ramp(_MUJOCO_BLUE, 0.0), 0.85)

    if self.border_width > 0.0:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -terrain_height / 2)
      border_inner_size = (
        np.maximum(1e-6, self.size[0] - 2 * self.border_width),
        np.maximum(1e-6, self.size[1] - 2 * self.border_width),
      )
      border_boxes = make_border(
        body, self.size, border_inner_size, terrain_height, border_center
      )
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=border_rgba))

    if self.add_floor:
      floor_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          (self.size[0] - 2 * self.border_width) / 2,
          (self.size[1] - 2 * self.border_width) / 2,
          0.05,
        ),
        pos=(self.size[0] / 2, self.size[1] / 2, -0.05),
      )
      geometries.append(TerrainGeometry(geom=floor_geom, color=(0.4, 0.4, 0.4, 1.0)))

    # Platform
    platform_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.platform_width / 2, self.platform_width / 2, terrain_height / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -terrain_height / 2),
    )
    geometries.append(TerrainGeometry(geom=platform_geom, color=(0.4, 0.4, 0.4, 1.0)))

    platform_half = self.platform_width / 2
    terrain_center = self.size[0] / 2
    platform_min = terrain_center - platform_half
    platform_max = terrain_center + platform_half

    for _ in range(num_boxes):
      # Random size.
      size_x = rng.uniform(*self.box_width_range)
      size_y = rng.uniform(*self.box_length_range)
      height = rng.uniform(*self.box_height_range)

      # Scale height by difficulty.
      height = height * (0.2 + 0.8 * difficulty)

      # Random position within inner area.
      pos_x = rng.uniform(
        self.border_width + size_x / 2, self.size[0] - self.border_width - size_x / 2
      )
      pos_y = rng.uniform(
        self.border_width + size_y / 2, self.size[1] - self.border_width - size_y / 2
      )

      # Avoid platform.
      if (platform_min - size_x / 2 <= pos_x <= platform_max + size_x / 2) and (
        platform_min - size_y / 2 <= pos_y <= platform_max + size_y / 2
      ):
        continue

      pos_z = height / 2

      # Random orientation (yaw).
      yaw = np.deg2rad(rng.uniform(*self.box_yaw_range))

      rgba = brand_ramp(_MUJOCO_BLUE, rng.uniform(0.3, 0.8))

      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, size_x / 2),
          np.maximum(1e-6, size_y / 2),
          np.maximum(1e-6, height / 2),
        ),
        pos=(pos_x, pos_y, pos_z),
      )
      # MuJoCo quat is (w, x, y, z).
      geom.quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
      geometries.append(TerrainGeometry(geom=geom, color=rgba))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxOpenStairsTerrainCfg(SubTerrainCfg):
  step_height_range: tuple[float, float] = (0.1, 0.2)
  step_width_range: tuple[float, float] = (0.4, 0.8)
  platform_width: float = 1.0
  border_width: float = 0.25
  step_thickness: float = 0.05
  inverted: bool = True

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng  # Unused.
    body = spec.body("terrain")
    geometries = []

    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )
    step_width = self.step_width_range[1] - difficulty * (
      self.step_width_range[1] - self.step_width_range[0]
    )

    # Compute number of steps.
    num_steps_x = int(
      (self.size[0] - 2 * self.border_width - self.platform_width) / (2 * step_width)
    )
    num_steps_y = int(
      (self.size[1] - 2 * self.border_width - self.platform_width) / (2 * step_width)
    )
    num_steps = int(min(num_steps_x, num_steps_y))

    first_step_rgba = brand_ramp(_MUJOCO_BLUE, 0.0)
    border_rgba = darken_rgba(first_step_rgba, 0.85)

    if self.border_width > 0.0:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -step_height / 2)
      border_inner_size = (
        self.size[0] - 2 * self.border_width,
        self.size[1] - 2 * self.border_width,
      )
      border_boxes = make_border(
        body, self.size, border_inner_size, step_height, border_center
      )
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=border_rgba))

    terrain_center = [0.5 * self.size[0], 0.5 * self.size[1], 0.0]
    terrain_size = (
      self.size[0] - 2 * self.border_width,
      self.size[1] - 2 * self.border_width,
    )

    rgba = brand_ramp(_MUJOCO_BLUE, 0.5)
    for k in range(num_steps):
      t = k / max(num_steps - 1, 1)
      rgba = brand_ramp(_MUJOCO_BLUE, t)

      box_size = (
        terrain_size[0] - 2 * k * step_width,
        terrain_size[1] - 2 * k * step_width,
      )

      # Inverted: Outer steps (small k) are higher (Bowl).
      # Normal: Inner steps (large k) are higher (Pyramid).
      if self.inverted:
        # Highest step (k=0) top is at 0 (border height).
        z_pos = -k * step_height - 0.5 * self.step_thickness
      else:
        # Lowest step (k=0) top is at step_height.
        z_pos = (k + 1) * step_height - 0.5 * self.step_thickness

      box_offset = (k + 0.5) * step_width

      # Top.
      box_pos = (
        terrain_center[0],
        terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
        z_pos,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, box_size[0] / 2.0),
          np.maximum(1e-6, step_width / 2.0),
          self.step_thickness / 2.0,
        ),
        pos=box_pos,
      )
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Bottom.
      box_pos = (
        terrain_center[0],
        terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
        z_pos,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, box_size[0] / 2.0),
          np.maximum(1e-6, step_width / 2.0),
          self.step_thickness / 2.0,
        ),
        pos=box_pos,
      )
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Right.
      box_pos = (
        terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
        terrain_center[1],
        z_pos,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, step_width / 2.0),
          np.maximum(1e-6, (box_size[1] - 2 * step_width) / 2.0),
          self.step_thickness / 2.0,
        ),
        pos=box_pos,
      )
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Left.
      box_pos = (
        terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
        terrain_center[1],
        z_pos,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, step_width / 2.0),
          np.maximum(1e-6, (box_size[1] - 2 * step_width) / 2.0),
          self.step_thickness / 2.0,
        ),
        pos=box_pos,
      )
      geometries.append(TerrainGeometry(geom=box, color=rgba))

    # Platform.
    platform_size = (
      np.maximum(1e-6, terrain_size[0] - 2 * num_steps * step_width),
      np.maximum(1e-6, terrain_size[1] - 2 * num_steps * step_width),
    )
    # Bowl: Align bottom (ground level).
    # Bowl: Highest step is border level. Platform is at bottom.
    # Pyramid: Align top-most level.
    if self.inverted:
      platform_h_center = -num_steps * step_height - 0.5 * self.step_thickness
    else:
      platform_h_center = (num_steps + 1) * step_height - 0.5 * self.step_thickness

    platform_pos = (terrain_center[0], terrain_center[1], platform_h_center)
    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(platform_size[0] / 2.0, platform_size[1] / 2.0, self.step_thickness / 2.0),
      pos=platform_pos,
    )
    geometries.append(TerrainGeometry(geom=box, color=rgba))

    origin = np.array(
      [
        terrain_center[0],
        terrain_center[1],
        platform_h_center + self.step_thickness / 2.0,
      ]
    )
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxRandomStairsTerrainCfg(SubTerrainCfg):
  step_width: float = 0.8
  step_height_range: tuple[float, float] = (0.1, 0.3)
  platform_width: float = 1.0
  border_width: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    # Compute number of steps.
    num_steps_x = int(
      (self.size[0] - 2 * self.border_width - self.platform_width)
      / (2 * self.step_width)
    )
    num_steps_y = int(
      (self.size[1] - 2 * self.border_width - self.platform_width)
      / (2 * self.step_width)
    )
    num_steps = int(min(num_steps_x, num_steps_y))

    first_step_rgba = brand_ramp(_MUJOCO_BLUE, 0.0)
    border_rgba = darken_rgba(first_step_rgba, 0.85)

    if self.border_width > 0.0:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -0.05)
      border_inner_size = (
        self.size[0] - 2 * self.border_width,
        self.size[1] - 2 * self.border_width,
      )
      border_boxes = make_border(body, self.size, border_inner_size, 0.1, border_center)
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=border_rgba))

    terrain_center = [0.5 * self.size[0], 0.5 * self.size[1], 0.0]
    terrain_size = (
      self.size[0] - 2 * self.border_width,
      self.size[1] - 2 * self.border_width,
    )

    rgba = brand_ramp(_MUJOCO_BLUE, 0.5)
    current_z = 0.0
    for k in range(num_steps):
      t = k / max(num_steps - 1, 1)
      rgba = brand_ramp(_MUJOCO_BLUE, t)

      h_low, h_high = self.step_height_range
      step_h = rng.uniform(h_low, h_high) * (0.5 + 0.5 * difficulty)
      total_h = current_z + step_h

      box_size = (
        terrain_size[0] - 2 * k * self.step_width,
        terrain_size[1] - 2 * k * self.step_width,
      )

      z_pos = total_h / 2
      box_offset = (k + 0.5) * self.step_width

      # Top.
      box_pos = (
        terrain_center[0],
        terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
        z_pos,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, box_size[0] / 2.0),
          np.maximum(1e-6, self.step_width / 2.0),
          np.maximum(1e-6, total_h / 2.0),
        ),
        pos=box_pos,
      )
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Bottom.
      box_pos = (
        terrain_center[0],
        terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
        z_pos,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, box_size[0] / 2.0),
          np.maximum(1e-6, self.step_width / 2.0),
          np.maximum(1e-6, total_h / 2.0),
        ),
        pos=box_pos,
      )
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Right.
      box_pos = (
        terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
        terrain_center[1],
        z_pos,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, self.step_width / 2.0),
          np.maximum(1e-6, (box_size[1] - 2 * self.step_width) / 2.0),
          np.maximum(1e-6, total_h / 2.0),
        ),
        pos=box_pos,
      )
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Left.
      box_pos = (
        terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
        terrain_center[1],
        z_pos,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, self.step_width / 2.0),
          np.maximum(1e-6, (box_size[1] - 2 * self.step_width) / 2.0),
          np.maximum(1e-6, total_h / 2.0),
        ),
        pos=box_pos,
      )
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      current_z = total_h

    # Platform
    platform_size = (
      np.maximum(1e-6, terrain_size[0] - 2 * num_steps * self.step_width),
      np.maximum(1e-6, terrain_size[1] - 2 * num_steps * self.step_width),
    )
    platform_pos = (terrain_center[0], terrain_center[1], current_z / 2)
    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(
        platform_size[0] / 2.0,
        platform_size[1] / 2.0,
        np.maximum(1e-6, current_z / 2.0),
      ),
      pos=platform_pos,
    )
    geometries.append(TerrainGeometry(geom=box, color=rgba))

    origin = np.array([terrain_center[0], terrain_center[1], current_z])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxSteppingStonesTerrainCfg(SubTerrainCfg):
  stone_size_range: tuple[float, float] = (0.4, 0.8)
  stone_distance_range: tuple[float, float] = (0.2, 0.5)
  stone_height: float = 0.2
  stone_height_variation: float = 0.1
  stone_size_variation: float = 0.1
  floor_depth: float = 2.0
  displacement_range: float = 0.1
  platform_width: float = 1.0
  border_width: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    # Difficulty-scaled variations.
    stone_size_variation = self.stone_size_variation * difficulty
    displacement_range = self.displacement_range * difficulty
    stone_height_variation = self.stone_height_variation * difficulty

    # Increase distance between stones with difficulty.
    d_low, d_high = self.stone_distance_range
    avg_distance = (d_low + d_high) / 2 + difficulty * (d_high - d_low) / 2

    # Grid spacing (stone size + gap).
    avg_stone_size = (self.stone_size_range[0] + self.stone_size_range[1]) / 2
    spacing = avg_stone_size + avg_distance

    # Aggressive grid density to reach borders.
    inner_w = self.size[0] - 2 * self.border_width
    inner_h = self.size[1] - 2 * self.border_width
    num_x = int(np.floor(inner_w / spacing)) + 1
    num_y = int(np.floor(inner_h / spacing)) + 1

    offset_x = self.border_width + (inner_w - (num_x - 1) * spacing) / 2
    offset_y = self.border_width + (inner_h - (num_y - 1) * spacing) / 2

    border_rgba = darken_rgba(brand_ramp(_MUJOCO_GREEN, 0.0), 0.85)
    z_center = (self.stone_height - self.floor_depth) / 2
    half_height = (self.stone_height + self.floor_depth) / 2

    if self.border_width > 0.0:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], z_center)
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        self.stone_height + self.floor_depth,
        border_center,
      )
      for b_geom in border_boxes:
        geometries.append(TerrainGeometry(geom=b_geom, color=border_rgba))

    # Ground floor (deep pit).
    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=(0.1, 0.1, 0.1, 1.0)))

    # Platform Column.
    platform_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(
        np.maximum(1e-6, self.platform_width / 2),
        np.maximum(1e-6, self.platform_width / 2),
        np.maximum(1e-6, half_height),
      ),
      pos=(self.size[0] / 2, self.size[1] / 2, z_center),
    )
    geometries.append(
      TerrainGeometry(geom=platform_geom, color=brand_ramp(_MUJOCO_GREEN, 0.5))
    )

    platform_half = self.platform_width / 2
    terrain_center = self.size[0] / 2
    platform_min = terrain_center - platform_half
    platform_max = terrain_center + platform_half

    inner_min_x, inner_max_x = self.border_width, self.size[0] - self.border_width
    inner_min_y, inner_max_y = self.border_width, self.size[1] - self.border_width

    for i in range(num_x):
      for j in range(num_y):
        avg_s_min, avg_s_max = self.stone_size_range
        base_size = (avg_s_min + avg_s_max) / 2

        # Proposed position with displacement.
        px = (
          offset_x + i * spacing + rng.uniform(-displacement_range, displacement_range)
        )
        py = (
          offset_y + j * spacing + rng.uniform(-displacement_range, displacement_range)
        )

        # Randomized size.
        size_x = base_size + rng.uniform(-stone_size_variation, stone_size_variation)
        size_y = base_size + rng.uniform(-stone_size_variation, stone_size_variation)

        # Initial bounds.
        x_min, x_max = px - size_x / 2, px + size_x / 2
        y_min, y_max = py - size_y / 2, py + size_y / 2

        # Avoid platform.
        margin = max(avg_s_max, self.stone_size_variation * 2) / 2
        if (platform_min - margin <= px <= platform_max + margin) and (
          platform_min - margin <= py <= platform_max + margin
        ):
          continue

        # Final clip against border.
        x_min = np.clip(x_min, inner_min_x, inner_max_x)
        x_max = np.clip(x_max, inner_min_x, inner_max_x)
        y_min = np.clip(y_min, inner_min_y, inner_max_y)
        y_max = np.clip(y_max, inner_min_y, inner_max_y)

        clipped_sx = x_max - x_min
        clipped_sy = y_max - y_min
        clipped_px = (x_min + x_max) / 2
        clipped_py = (y_min + y_max) / 2

        if clipped_sx < 0.05 or clipped_sy < 0.05:
          continue

        # Stones grow from the floor up to around ground level.
        h = (
          self.floor_depth
          + self.stone_height
          + rng.uniform(-stone_height_variation, stone_height_variation)
        )
        pos_z = -self.floor_depth + h / 2

        rgba = brand_ramp(_MUJOCO_GREEN, rng.uniform(0.4, 0.7))

        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(
            np.maximum(1e-6, clipped_sx / 2),
            np.maximum(1e-6, clipped_sy / 2),
            np.maximum(1e-6, h / 2),
          ),
          pos=(clipped_px, clipped_py, pos_z),
        )
        geometries.append(TerrainGeometry(geom=geom, color=rgba))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, self.stone_height])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxNarrowBeamsTerrainCfg(SubTerrainCfg):
  num_beams: int = 16
  beam_width_range: tuple[float, float] = (0.2, 0.4)
  beam_height: float = 0.2
  spacing: float = 0.8
  platform_width: float = 1.0
  border_width: float = 0.25
  floor_depth: float = 2.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    # Number of beams can be increased with difficulty if desired.
    num_beams = int(self.num_beams)

    # Narrower beams with difficulty (interp from max to min).
    w_min, w_max = self.beam_width_range
    beam_width = w_max - difficulty * (w_max - w_min)

    border_rgba = darken_rgba(brand_ramp(_MUJOCO_BLUE, 0.0), 0.85)
    z_center = (self.beam_height - self.floor_depth) / 2
    half_height = (self.beam_height + self.floor_depth) / 2

    if self.border_width > 0.0:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], z_center)
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        self.beam_height + self.floor_depth,
        border_center,
      )
      for b_geom in border_boxes:
        geometries.append(TerrainGeometry(geom=b_geom, color=border_rgba))

    # Ground floor (deep pit).
    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=(0.1, 0.1, 0.1, 1.0)))

    # Platform Column. Top at beam_height.
    platform_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(
        np.maximum(1e-6, self.platform_width / 2),
        np.maximum(1e-6, self.platform_width / 2),
        np.maximum(1e-6, half_height),
      ),
      pos=(self.size[0] / 2, self.size[1] / 2, z_center),
    )
    geometries.append(
      TerrainGeometry(geom=platform_geom, color=brand_ramp(_MUJOCO_BLUE, 0.5))
    )

    inner_size = self.size[0] - 2 * self.border_width
    center_x, center_y = self.size[0] / 2, self.size[1] / 2
    platform_radius = self.platform_width / 2

    # Radial beams as columns.
    angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
    for angle in angles:
      # Distance to the square border at this angle.
      # r = (L/2) / max(|cos(theta)|, |sin(theta)|)
      cos_a = abs(np.cos(angle))
      sin_a = abs(np.sin(angle))
      dist_to_border = (inner_size / 2) / max(cos_a, sin_a)

      # Additional length so that the farthest corner reaches the border.
      # delta_L = (w/2) * tan(local_theta) = (w/2) * min(cos, sin) / max(cos, sin)
      extra_length = (beam_width / 2) * min(cos_a, sin_a) / max(cos_a, sin_a)

      beam_length = dist_to_border + extra_length - platform_radius

      dist_to_center = platform_radius + beam_length / 2
      px = center_x + dist_to_center * np.cos(angle)
      py = center_y + dist_to_center * np.sin(angle)
      pz = z_center

      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
          np.maximum(1e-6, beam_length / 2),
          np.maximum(1e-6, beam_width / 2),
          np.maximum(1e-6, half_height),
        ),
        pos=(px, py, pz),
      )
      # Quat for yaw = angle.
      geom.quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
      geometries.append(TerrainGeometry(geom=geom, color=brand_ramp(_MUJOCO_BLUE, 0.5)))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, self.beam_height])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxTiltedGridTerrainCfg(SubTerrainCfg):
  grid_width: float = 1.0
  tilt_range_deg: float = 15.0
  height_range: float = 0.1
  platform_width: float = 1.0
  border_width: float = 0.25
  floor_depth: float = 2.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    # Difficulty-scaled variations.
    max_tilt = np.deg2rad(self.tilt_range_deg * difficulty)
    actual_range = self.height_range * difficulty

    # Logic adopted from BoxRandomGridTerrainCfg for consistent alignment.
    num_boxes_x = int((self.size[0] - 2 * self.border_width) / self.grid_width)
    num_boxes_y = int((self.size[1] - 2 * self.border_width) / self.grid_width)

    border_actual = self.size[0] - num_boxes_x * self.grid_width
    half_border = border_actual / 2

    border_rgba = darken_rgba(brand_ramp(_MUJOCO_GREEN, 0.0), 0.85)
    base_h = 0.2
    z_center = (base_h - self.floor_depth) / 2
    half_height = (base_h + self.floor_depth) / 2

    # Border.
    if border_actual > 0:
      border_center = (self.size[0] / 2, self.size[1] / 2, z_center)
      border_boxes = make_border(
        body,
        self.size,
        (num_boxes_x * self.grid_width, num_boxes_y * self.grid_width),
        base_h + self.floor_depth,
        border_center,
      )
      for b_geom in border_boxes:
        geometries.append(TerrainGeometry(geom=b_geom, color=border_rgba))

    # Ground floor.
    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=(0.1, 0.1, 0.1, 1.0)))

    # Platform.
    platform_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.platform_width / 2, self.platform_width / 2, half_height),
      pos=(self.size[0] / 2, self.size[1] / 2, z_center),
    )
    geometries.append(
      TerrainGeometry(geom=platform_geom, color=brand_ramp(_MUJOCO_GREEN, 0.5))
    )

    platform_half = self.platform_width / 2
    terrain_center = self.size[0] / 2
    platform_min = terrain_center - platform_half
    platform_max = terrain_center + platform_half

    for i in range(num_boxes_x):
      bx_center = half_border + (i + 0.5) * self.grid_width
      for j in range(num_boxes_y):
        by_center = half_border + (j + 0.5) * self.grid_width

        # Skip if inside platform.
        if (platform_min <= bx_center <= platform_max) and (
          platform_min <= by_center <= platform_max
        ):
          continue

        h_noise = rng.uniform(-actual_range / 2, actual_range / 2)
        total_h = base_h + h_noise

        tilt_x = rng.uniform(-max_tilt, max_tilt)
        tilt_y = rng.uniform(-max_tilt, max_tilt)

        # Mesh vertices.
        # Vertical sides: top and bottom x,y are identical.
        x_min, x_max = bx_center - self.grid_width / 2, bx_center + self.grid_width / 2
        y_min, y_max = by_center - self.grid_width / 2, by_center + self.grid_width / 2

        verts = []
        # Bottom verts 0-3 (at -floor_depth)
        for vx in [x_min, x_max]:
          for vy in [y_min, y_max]:
            verts.append([vx, vy, -self.floor_depth])
        # Top verts 4-7 (tilted)
        for vx in [x_min, x_max]:
          for vy in [y_min, y_max]:
            vz = total_h + tilt_x * (vx - bx_center) + tilt_y * (vy - by_center)
            verts.append([vx, vy, vz])

        # Faces ccw from outside.
        # 0:(min,min), 1:(min,max), 2:(max,min), 3:(max,max)
        # 4-7 are same x,y as 0-3 but at top.
        # fmt: off
        faces = [
          4, 6, 7, 4, 7, 5,  # Top (+z)
          0, 1, 3, 0, 3, 2,  # Bottom (-z)
          0, 2, 6, 0, 6, 4,  # Front (-y)
          1, 5, 7, 1, 7, 3,  # Back (+y)
          0, 4, 5, 0, 5, 1,  # Left (-x)
          2, 3, 7, 2, 7, 6,  # Right (+x)
        ]
        # fmt: on

        m_name = f"tile_{i}_{j}_{rng.integers(int(1e9))}"
        mesh = spec.add_mesh(
          name=m_name,
          uservert=np.array(verts).flatten().tolist(),
          userface=np.array(faces).flatten().tolist(),
        )

        rgba = brand_ramp(_MUJOCO_GREEN, rng.uniform(0.3, 0.7))
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_MESH,
          meshname=mesh.name,
          pos=(0, 0, 0),
        )
        geometries.append(TerrainGeometry(geom=geom, color=rgba))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, base_h])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxNestedRingsTerrainCfg(SubTerrainCfg):
  num_rings: int = 5
  ring_width_range: tuple[float, float] = (0.3, 0.6)
  gap_range: tuple[float, float] = (0.0, 0.2)
  height_range: tuple[float, float] = (0.1, 0.4)
  platform_width: float = 1.0
  border_width: float = 0.25
  floor_depth: float = 2.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    # Difficulty scaling: wider width range and higher average height.
    h_scale = 1.0 + difficulty * 0.5
    w_min, w_max = self.ring_width_range
    ring_width = w_max - difficulty * (w_max - w_min)

    border_rgba = darken_rgba(brand_ramp(_MUJOCO_BLUE, 0.0), 0.85)
    # Use ground level z=0 as top of border/beams for consistency with NarrowBeams.
    # In beam terrain, border top was at beam_height.

    if self.border_width > 0.0:
      border_h = 0.5
      border_center = (
        0.5 * self.size[0],
        0.5 * self.size[1],
        (border_h - self.floor_depth) / 2,
      )
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        border_h + self.floor_depth,
        border_center,
      )
      for b_geom in border_boxes:
        geometries.append(TerrainGeometry(geom=b_geom, color=border_rgba))

    # Ground floor (deep pit).
    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=(0.1, 0.1, 0.1, 1.0)))

    terrain_center = [0.5 * self.size[0], 0.5 * self.size[1], 0.0]
    terrain_size = (
      self.size[0] - 2 * self.border_width,
      self.size[1] - 2 * self.border_width,
    )

    current_outer_size = list(terrain_size)

    gap_min, gap_max = self.gap_range
    gap = gap_min + difficulty * (gap_max - gap_min)

    for k in range(self.num_rings):
      # Ring k: randomized height.
      h = rng.uniform(self.height_range[0], self.height_range[1]) * h_scale

      t = k / max(self.num_rings - 1, 1)
      rgba = brand_ramp(_MUJOCO_BLUE, t)

      # Outer dimensions of this ring.
      ring_outer_size = (
        current_outer_size[0],
        current_outer_size[1],
      )

      # Stop if we get too small.
      if (
        ring_outer_size[0] <= self.platform_width
        or ring_outer_size[1] <= self.platform_width
      ):
        break

      # Position each ring segment based on current_outer_size.

      half_h = (h + self.floor_depth) / 2
      z_pos = (h - self.floor_depth) / 2

      # Four segments for the ring.
      # Top/Bottom
      for dy in [-1, 1]:
        box_pos = (
          terrain_center[0],
          terrain_center[1] + dy * (ring_outer_size[1] - ring_width) / 2,
          z_pos,
        )
        box = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(
            np.maximum(1e-6, ring_outer_size[0] / 2.0),
            np.maximum(1e-6, ring_width / 2.0),
            half_h,
          ),
          pos=box_pos,
        )
        geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Left/Right
      for dx in [-1, 1]:
        box_pos = (
          terrain_center[0] + dx * (ring_outer_size[0] - ring_width) / 2,
          terrain_center[1],
          z_pos,
        )
        box = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(
            np.maximum(1e-6, ring_width / 2.0),
            np.maximum(1e-6, (ring_outer_size[1] - 2 * ring_width) / 2.0),
            half_h,
          ),
          pos=box_pos,
        )
        geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Shrink current_outer_size for next ring.
      current_outer_size[0] -= 2 * (ring_width + gap)
      current_outer_size[1] -= 2 * (ring_width + gap)

    # Center Platform Column matches the remaining hole exactly.
    platform_size = (
      np.maximum(
        1e-2, current_outer_size[0] + 2 * gap
      ),  # Fill the ring hole + gap area.
      np.maximum(1e-2, current_outer_size[1] + 2 * gap),
    )
    platform_h = 0.2

    platform_half_h = (platform_h + self.floor_depth) / 2
    platform_z = (platform_h - self.floor_depth) / 2

    platform_pos = (terrain_center[0], terrain_center[1], platform_z)
    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(platform_size[0] / 2.0, platform_size[1] / 2.0, platform_half_h),
      pos=platform_pos,
    )
    geometries.append(TerrainGeometry(geom=box, color=brand_ramp(_MUJOCO_BLUE, 0.5)))

    origin = np.array([terrain_center[0], terrain_center[1], platform_h])
    return TerrainOutput(origin=origin, geometries=geometries)
