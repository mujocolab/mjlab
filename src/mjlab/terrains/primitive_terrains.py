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
    num_steps_x = (self.size[0] - 2 * self.border_width - self.platform_width) // (
      2 * self.step_width
    ) + 1
    num_steps_y = (self.size[1] - 2 * self.border_width - self.platform_width) // (
      2 * self.step_width
    ) + 1
    num_steps = int(min(num_steps_x, num_steps_y))

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

      # Top.
      box_pos = (
        terrain_center[0],
        terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
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
        size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
        pos=box_pos,
      )
      boxes.append(box)

      if self.holes:
        box_dims = (self.step_width, box_size[1], box_height)
      else:
        box_dims = (self.step_width, box_size[1] - 2 * self.step_width, box_height)

      # Right.
      box_pos = (
        terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
        terrain_center[1],
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
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
        size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
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
      size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
      pos=box_pos,
    )
    boxes.append(box)
    origin = np.array(
      [terrain_center[0], terrain_center[1], (num_steps + 1) * step_height]
    )
    platform_rgba = _get_platform_color(_MUJOCO_BLUE)
    box_colors.append(platform_rgba)

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
    num_steps_x = (self.size[0] - 2 * self.border_width - self.platform_width) // (
      2 * self.step_width
    ) + 1
    num_steps_y = (self.size[1] - 2 * self.border_width - self.platform_width) // (
      2 * self.step_width
    ) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
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

      # Top.
      box_pos = (
        terrain_center[0],
        terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
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
        size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
        pos=box_pos,
      )
      boxes.append(box)

      if self.holes:
        box_dims = (self.step_width, box_size[1], box_height)
      else:
        box_dims = (self.step_width, box_size[1] - 2 * self.step_width, box_height)

      # Right.
      box_pos = (
        terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
        terrain_center[1],
        box_z,
      )
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
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
        size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
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
      size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
      pos=box_pos,
    )
    boxes.append(box)
    origin = np.array(
      [terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height]
    )
    platform_rgba = _get_platform_color(_MUJOCO_RED)
    box_colors.append(platform_rgba)

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
  num_boxes: int = 250
  box_size_range: tuple[float, float] = (0.3, 1.0)
  box_height_range: tuple[float, float] = (0.1, 0.4)
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
        self.size[0] - 2 * self.border_width,
        self.size[1] - 2 * self.border_width,
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
    platform_rgba = _get_platform_color(_MUJOCO_BLUE)
    platform_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.platform_width / 2, self.platform_width / 2, terrain_height / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -terrain_height / 2),
    )
    geometries.append(TerrainGeometry(geom=platform_geom, color=platform_rgba))

    platform_half = self.platform_width / 2
    terrain_center = self.size[0] / 2
    platform_min = terrain_center - platform_half
    platform_max = terrain_center + platform_half

    for _ in range(num_boxes):
      # Random size.
      size_x = rng.uniform(*self.box_size_range)
      size_y = rng.uniform(*self.box_size_range)
      height = rng.uniform(*self.box_height_range)

      # Random position within inner area.
      inner_size_x = self.size[0] - 2 * self.border_width
      inner_size_y = self.size[1] - 2 * self.border_width
      pos_x = rng.uniform(self.border_width + size_x / 2, self.size[0] - self.border_width - size_x / 2)
      pos_y = rng.uniform(self.border_width + size_y / 2, self.size[1] - self.border_width - size_y / 2)
      
      # Avoid platform.
      if (platform_min - size_x / 2 <= pos_x <= platform_max + size_x / 2) and \
         (platform_min - size_y / 2 <= pos_y <= platform_max + size_y / 2):
        continue

      pos_z = height / 2

      # Random orientation (yaw).
      yaw = np.deg2rad(rng.uniform(*self.box_yaw_range))

      rgba = brand_ramp(_MUJOCO_BLUE, rng.uniform(0.3, 0.8))

      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(size_x / 2, size_y / 2, height / 2),
        pos=(pos_x, pos_y, pos_z),
      )
      # MuJoCo quat is (w, x, y, z).
      geom.quat = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
      geometries.append(TerrainGeometry(geom=geom, color=rgba))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxOpenStairsTerrainCfg(SubTerrainCfg):
  step_height_range: tuple[float, float] = (0.1, 0.2)
  step_width: float = 0.4
  platform_width: float = 1.0
  border_width: float = 0.25
  step_thickness: float = 0.05

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng  # Unused.
    body = spec.body("terrain")
    geometries = []

    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )

    # Compute number of steps.
    num_steps_x = (self.size[0] - 2 * self.border_width - self.platform_width) // (
      2 * self.step_width
    ) + 1
    num_steps_y = (self.size[1] - 2 * self.border_width - self.platform_width) // (
      2 * self.step_width
    ) + 1
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

    for k in range(num_steps):
      t = k / max(num_steps - 1, 1)
      rgba = brand_ramp(_MUJOCO_BLUE, t)

      box_size = (
        terrain_size[0] - 2 * k * self.step_width,
        terrain_size[1] - 2 * k * self.step_width,
      )
      
      z_pos = (k + 0.5) * step_height
      box_offset = (k + 0.5) * self.step_width
      
      # Top.
      box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, z_pos)
      box = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(box_size[0] / 2.0, self.step_width / 2.0, self.step_thickness / 2.0), pos=box_pos)
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Bottom.
      box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, z_pos)
      box = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(box_size[0] / 2.0, self.step_width / 2.0, self.step_thickness / 2.0), pos=box_pos)
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Right.
      box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], z_pos)
      box = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(self.step_width / 2.0, (box_size[1] - 2 * self.step_width) / 2.0, self.step_thickness / 2.0), pos=box_pos)
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Left.
      box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], z_pos)
      box = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(self.step_width / 2.0, (box_size[1] - 2 * self.step_width) / 2.0, self.step_thickness / 2.0), pos=box_pos)
      geometries.append(TerrainGeometry(geom=box, color=rgba))

    # Platform
    platform_size = (
      terrain_size[0] - 2 * num_steps * self.step_width,
      terrain_size[1] - 2 * num_steps * self.step_width,
    )
    platform_h = (num_steps + 1) * step_height
    platform_pos = (terrain_center[0], terrain_center[1], num_steps * step_height / 2)
    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(platform_size[0] / 2.0, platform_size[1] / 2.0, platform_h / 2.0),
      pos=platform_pos,
    )
    platform_rgba = _get_platform_color(_MUJOCO_BLUE)
    geometries.append(TerrainGeometry(geom=box, color=platform_rgba))

    origin = np.array([terrain_center[0], terrain_center[1], num_steps * step_height])
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
    num_steps_x = (self.size[0] - 2 * self.border_width - self.platform_width) // (
      2 * self.step_width
    ) + 1
    num_steps_y = (self.size[1] - 2 * self.border_width - self.platform_width) // (
      2 * self.step_width
    ) + 1
    num_steps = int(min(num_steps_x, num_steps_y))

    first_step_rgba = brand_ramp(_MUJOCO_BLUE, 0.0)
    border_rgba = darken_rgba(first_step_rgba, 0.85)

    if self.border_width > 0.0:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -0.05)
      border_inner_size = (
        self.size[0] - 2 * self.border_width,
        self.size[1] - 2 * self.border_width,
      )
      border_boxes = make_border(
        body, self.size, border_inner_size, 0.1, border_center
      )
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=border_rgba))

    terrain_center = [0.5 * self.size[0], 0.5 * self.size[1], 0.0]
    terrain_size = (
      self.size[0] - 2 * self.border_width,
      self.size[1] - 2 * self.border_width,
    )

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
      
      # For solid staircase, we can use a single box that is high enough for each "ring".
      # But to correctly follow the "random height" requirement per step, 
      # we should use 4 boxes per level.
      
      # Top.
      box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, z_pos)
      box = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(box_size[0] / 2.0, self.step_width / 2.0, total_h / 2.0), pos=box_pos)
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Bottom.
      box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, z_pos)
      box = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(box_size[0] / 2.0, self.step_width / 2.0, total_h / 2.0), pos=box_pos)
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Right.
      box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], z_pos)
      box = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(self.step_width / 2.0, (box_size[1] - 2 * self.step_width) / 2.0, total_h / 2.0), pos=box_pos)
      geometries.append(TerrainGeometry(geom=box, color=rgba))

      # Left.
      box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], z_pos)
      box = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(self.step_width / 2.0, (box_size[1] - 2 * self.step_width) / 2.0, total_h / 2.0), pos=box_pos)
      geometries.append(TerrainGeometry(geom=box, color=rgba))
      
      current_z = total_h

    # Platform
    platform_size = (
      terrain_size[0] - 2 * num_steps * self.step_width,
      terrain_size[1] - 2 * num_steps * self.step_width,
    )
    platform_pos = (terrain_center[0], terrain_center[1], current_z / 2)
    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(platform_size[0] / 2.0, platform_size[1] / 2.0, current_z / 2.0),
      pos=platform_pos,
    )
    platform_rgba = _get_platform_color(_MUJOCO_BLUE)
    geometries.append(TerrainGeometry(geom=box, color=platform_rgba))

    origin = np.array([terrain_center[0], terrain_center[1], current_z])
    return TerrainOutput(origin=origin, geometries=geometries)

@dataclass(kw_only=True)
class BoxSteppingStonesTerrainCfg(SubTerrainCfg):
  stone_size_range: tuple[float, float] = (0.4, 0.8)
  stone_distance_range: tuple[float, float] = (0.2, 0.5)
  stone_height: float = 0.5
  stone_height_variation: float = 0.1
  platform_width: float = 1.0
  border_width: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    # Increase distance between stones with difficulty.
    d_low, d_high = self.stone_distance_range
    avg_distance = (d_low + d_high) / 2 + difficulty * (d_high - d_low) / 2
    
    # Grid spacing (stone size + gap).
    avg_stone_size = (self.stone_size_range[0] + self.stone_size_range[1]) / 2
    spacing = avg_stone_size + avg_distance

    num_x = int((self.size[0] - 2 * self.border_width) / spacing)
    num_y = int((self.size[1] - 2 * self.border_width) / spacing)

    offset_x = self.border_width + (self.size[0] - 2 * self.border_width - (num_x - 1) * spacing) / 2
    offset_y = self.border_width + (self.size[1] - 2 * self.border_width - (num_y - 1) * spacing) / 2

    border_rgba = darken_rgba(brand_ramp(_MUJOCO_GREEN, 0.0), 0.85)
    if self.border_width > 0.0:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -0.05)
      border_boxes = make_border(body, self.size, (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width), 0.1, border_center)
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=border_rgba))

    # Platform.
    platform_rgba = _get_platform_color(_MUJOCO_GREEN)
    platform_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.platform_width / 2, self.platform_width / 2, self.stone_height / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, self.stone_height / 2),
    )
    geometries.append(TerrainGeometry(geom=platform_geom, color=platform_rgba))

    platform_half = self.platform_width / 2
    terrain_center = self.size[0] / 2
    platform_min = terrain_center - platform_half
    platform_max = terrain_center + platform_half

    for i in range(num_x):
      for j in range(num_y):
        # Randomize size.
        size_x = rng.uniform(*self.stone_size_range)
        size_y = rng.uniform(*self.stone_size_range)
        
        px = offset_x + i * spacing + rng.uniform(-0.1, 0.1)
        py = offset_y + j * spacing + rng.uniform(-0.1, 0.1)
        
        # Avoid platform.
        if (platform_min - size_x / 2 <= px <= platform_max + size_x / 2) and \
           (platform_min - size_y / 2 <= py <= platform_max + size_y / 2):
          continue

        h = self.stone_height + rng.uniform(-self.stone_height_variation, self.stone_height_variation)
        h = h * (0.5 + 0.5 * difficulty)
        
        rgba = brand_ramp(_MUJOCO_GREEN, rng.uniform(0.4, 0.7))
        
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(size_x / 2, size_y / 2, h / 2),
          pos=(px, py, h / 2),
        )
        geometries.append(TerrainGeometry(geom=geom, color=rgba))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, self.stone_height])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxNarrowBeamsTerrainCfg(SubTerrainCfg):
  num_beams: int = 3
  beam_width: float = 0.1
  beam_height: float = 0.2
  spacing: float = 0.8
  platform_width: float = 1.0
  border_width: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng  # Unused.
    body = spec.body("terrain")
    geometries = []

    # Narrower beams with difficulty.
    beam_width = self.beam_width / (0.5 + 0.5 * difficulty)

    border_rgba = darken_rgba(brand_ramp(_MUJOCO_BLUE, 0.0), 0.85)
    if self.border_width > 0.0:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -0.05)
      border_boxes = make_border(body, self.size, (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width), 0.1, border_center)
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=border_rgba))

    # Platform.
    platform_rgba = _get_platform_color(_MUJOCO_BLUE)
    platform_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.platform_width / 2, self.platform_width / 2, self.beam_height / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, self.beam_height / 2),
    )
    geometries.append(TerrainGeometry(geom=platform_geom, color=platform_rgba))

    inner_size_x = self.size[0] - 2 * self.border_width
    beam_length = (inner_size_x - self.platform_width) / 2
    
    total_y_width = (self.num_beams - 1) * self.spacing
    start_y = (self.size[1] - total_y_width) / 2

    for i in range(self.num_beams):
      y_pos = start_y + i * self.spacing
      z_pos = self.beam_height / 2
      
      # Beam X.
      for side in [-1, 1]:
        x_pos = self.size[0] / 2 + side * (self.platform_width / 2 + beam_length / 2)
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(beam_length / 2, beam_width / 2, self.beam_height / 2),
          pos=(x_pos, y_pos, z_pos),
        )
        geometries.append(TerrainGeometry(geom=geom, color=brand_ramp(_MUJOCO_BLUE, 0.5)))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, self.beam_height])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxTiltedGridTerrainCfg(SubTerrainCfg):
  grid_width: float = 1.0
  tilt_range_deg: float = 15.0
  height_range: tuple[float, float] = (0.0, 0.2)
  platform_width: float = 1.0
  border_width: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    num_x = int((self.size[0] - 2 * self.border_width) / self.grid_width)
    num_y = int((self.size[1] - 2 * self.border_width) / self.grid_width)
    
    offset_x = self.border_width + (self.size[0] - 2 * self.border_width - (num_x - 1) * self.grid_width) / 2
    offset_y = self.border_width + (self.size[1] - 2 * self.border_width - (num_y - 1) * self.grid_width) / 2

    border_rgba = darken_rgba(brand_ramp(_MUJOCO_GREEN, 0.0), 0.85)
    if self.border_width > 0.0:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -0.05)
      border_boxes = make_border(body, self.size, (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width), 0.1, border_center)
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=border_rgba))

    # Platform.
    platform_rgba = _get_platform_color(_MUJOCO_GREEN)
    platform_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.platform_width / 2, self.platform_width / 2, 0.5 / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, 0.5 / 2),
    )
    geometries.append(TerrainGeometry(geom=platform_geom, color=platform_rgba))

    platform_half = self.platform_width / 2
    terrain_center = self.size[0] / 2
    platform_min = terrain_center - platform_half
    platform_max = terrain_center + platform_half

    max_tilt = np.deg2rad(self.tilt_range_deg * difficulty)

    for i in range(num_x):
      for j in range(num_y):
        px = offset_x + i * self.grid_width
        py = offset_y + j * self.grid_width
        
        # Avoid platform.
        if (platform_min - self.grid_width / 2 <= px <= platform_max + self.grid_width / 2) and \
           (platform_min - self.grid_width / 2 <= py <= platform_max + self.grid_width / 2):
          continue

        h_noise = rng.uniform(*self.height_range)
        # Use a base height so that tilting doesn't make corners go below 0 easily.
        base_h = 0.5
        total_h = base_h + h_noise
        
        # Random tilt.
        tilt_x = rng.uniform(-max_tilt, max_tilt)
        tilt_y = rng.uniform(-max_tilt, max_tilt)
        
        rgba = brand_ramp(_MUJOCO_GREEN, rng.uniform(0.3, 0.7))
        
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(self.grid_width / 2, self.grid_width / 2, total_h / 2),
          pos=(px, py, total_h / 2),
        )
        # Convert Euler (tilt_x, tilt_y, 0) to Quat (w, x, y, z).
        cx = np.cos(tilt_x / 2)
        sx = np.sin(tilt_x / 2)
        cy = np.cos(tilt_y / 2)
        sy = np.sin(tilt_y / 2)
        geom.quat = (cx * cy, sx * cy, cx * sy, -sx * sy)
        geometries.append(TerrainGeometry(geom=geom, color=rgba))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.5])
    return TerrainOutput(origin=origin, geometries=geometries)
