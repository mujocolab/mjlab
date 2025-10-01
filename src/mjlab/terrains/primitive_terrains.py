# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
  step_height_range: tuple[float, float]
  step_width: float
  platform_width: float = 1.0
  holes: bool = False

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
  grid_height_range: tuple[float, float]
  platform_width: float = 1.0
  holes: bool = False
  merge_similar_heights: bool = False
  height_merge_threshold: float = 0.05
  max_merge_distance: int = 3

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

    num_boxes_x = int(self.size[0] / self.grid_width)
    num_boxes_y = int(self.size[1] / self.grid_width)

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
