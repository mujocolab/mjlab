from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.sub_terrain_cfg import SubTerrainCfg
from mjlab.terrains.utils import make_border

"""
NOTE: mujoco geoms expect half-lengths hence we divide all sizes by 2.
"""


def platform_from_base(base_rgb):
  r, g, b = base_rgb
  mx, mn = max(r, g, b), min(r, g, b)
  d = mx - mn
  if d == 0:
    h = 0.0
  elif mx == r:
    h = ((g - b) / d) % 6
  elif mx == g:
    h = (b - r) / d + 2
  else:
    h = (r - g) / d + 4
  h /= 6.0
  v = mx
  s = 0 if v == 0 else d / v

  # desaturate + lighten
  s = s * 0.4
  v = min(1.0, v + 0.25)

  # HSV→RGB
  i = int(h * 6)
  f = h * 6 - i
  p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
  i %= 6
  if i == 0:
    r2, g2, b2 = v, t, p
  elif i == 1:
    r2, g2, b2 = q, v, p
  elif i == 2:
    r2, g2, b2 = p, v, t
  elif i == 3:
    r2, g2, b2 = p, q, v
  elif i == 4:
    r2, g2, b2 = t, p, v
  else:
    r2, g2, b2 = v, p, q
  return (r2, g2, b2, 1.0)


def brand_ramp(base_rgb, t, alpha=1.0):
  """
  base_rgb: (r,g,b) in [0,1]
  t: 0..1 (step or normalized height)
  """
  r, g, b = base_rgb
  # convert to HSV
  mx, mn = max(r, g, b), min(r, g, b)
  d = mx - mn
  if d == 0:
    h = 0.0
  elif mx == r:
    h = ((g - b) / d) % 6
  elif mx == g:
    h = (b - r) / d + 2
  else:
    h = (r - g) / d + 4
  h /= 6.0
  v = mx
  s = 0.0 if v == 0 else d / v

  # ramp: slightly increase V and S with t, but clamp
  v = min(1.0, 0.75 + 0.25 * t)
  s = min(1.0, s * (0.85 + 0.25 * t))

  # HSV->RGB
  i = int(h * 6)
  f = h * 6 - i
  p = v * (1 - s)
  q = v * (1 - f * s)
  tcol = v * (1 - (1 - f) * s)
  i %= 6
  if i == 0:
    r2, g2, b2 = v, tcol, p
  elif i == 1:
    r2, g2, b2 = q, v, p
  elif i == 2:
    r2, g2, b2 = p, v, tcol
  elif i == 3:
    r2, g2, b2 = p, q, v
  elif i == 4:
    r2, g2, b2 = tcol, p, v
  else:
    r2, g2, b2 = v, p, q
  return (float(r2), float(g2), float(b2), float(alpha))


def darken_rgba(rgba, factor=0.85):
  r, g, b, a = rgba
  return (r * factor, g * factor, b * factor, a)


@dataclass(kw_only=True)
class BoxPyramidStairsTerrainCfg(SubTerrainCfg):
  """Configuration for a pyramid stairs terrain."""

  border_width: float = 0.0
  step_height_range: tuple[float, float]
  step_width: float
  platform_width: float = 1.0
  holes: bool = False

  def function(self, difficulty: float, spec: mujoco.MjSpec):
    boxes = []
    box_colors = []

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

    base_blue = (0.20, 0.45, 0.95)
    first_step_rgba = brand_ramp(base_blue, 0.0)
    border_rgba = darken_rgba(first_step_rgba, 0.85)

    if self.border_width > 0.0 and not self.holes:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -step_height / 2)
      border_inner_size = (
        self.size[0] - 2 * self.border_width,
        self.size[1] - 2 * self.border_width,
      )
      border_boxes = make_border(
        spec, self.size, border_inner_size, step_height, border_center
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
      rgba = brand_ramp(base_blue, t)
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
      box = spec.worldbody.add_geom(
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
      box = spec.worldbody.add_geom(
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
      box = spec.worldbody.add_geom(
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
      box = spec.worldbody.add_geom(
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
    box = spec.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
      pos=box_pos,
    )
    boxes.append(box)
    origin = np.array(
      [terrain_center[0], terrain_center[1], (num_steps + 1) * step_height]
    )
    platform_rgba = platform_from_base(base_blue)
    box_colors.append(platform_rgba)
    return origin, boxes, box_colors


@dataclass(kw_only=True)
class BoxInvertedPyramidStairsTerrainCfg(BoxPyramidStairsTerrainCfg):
  def function(self, difficulty: float, spec: mujoco.MjSpec):
    boxes = []
    box_colors = []

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

    base_red = (0.90, 0.30, 0.30)
    first_step_rgba = brand_ramp(base_red, 0.0)
    border_rgba = darken_rgba(first_step_rgba, 0.85)

    if self.border_width > 0.0 and not self.holes:
      border_center = (0.5 * self.size[0], 0.5 * self.size[1], -0.5 * step_height)
      border_inner_size = (
        self.size[0] - 2 * self.border_width,
        self.size[1] - 2 * self.border_width,
      )
      border_boxes = make_border(
        spec, self.size, border_inner_size, step_height, border_center
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
      rgba = brand_ramp(base_red, t)
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
      box = spec.worldbody.add_geom(
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
      box = spec.worldbody.add_geom(
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
      box = spec.worldbody.add_geom(
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
      box = spec.worldbody.add_geom(
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
    box = spec.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
      pos=box_pos,
    )
    boxes.append(box)
    origin = np.array(
      [terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height]
    )
    # box_colors.append((0.5, 0.7, 1.0, 1.0))
    # platform_rgba = (0.72, 0.72, 0.72, 1.0)
    platform_rgba = platform_from_base(base_red)
    box_colors.append(platform_rgba)
    return origin, boxes, box_colors


@dataclass(kw_only=True)
class BoxRandomGridTerrainCfg(SubTerrainCfg):
  grid_width: float
  grid_height_range: tuple[float, float]
  platform_width: float = 1.0
  holes: bool = False
  merge_similar_heights: bool = True  # New option to enable merging
  height_merge_threshold: float = 0.05  # Heights within this range get merged
  max_merge_distance: int = 3  # Maximum grid cells to merge

  def function(self, difficulty: float, spec: mujoco.MjSpec):
    if self.size[0] != self.size[1]:
      raise ValueError(f"The terrain must be square. Received size: {self.size}.")

    grid_height = self.grid_height_range[0] + difficulty * (
      self.grid_height_range[1] - self.grid_height_range[0]
    )

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

    base_green = (0.25, 0.80, 0.45)
    first_step_rgba = brand_ramp(base_green, 0.0)
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
      box = spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
      )
      boxes_list.append(box)
      box_colors.append(border_rgba)

    height_map = np.random.uniform(
      -grid_height, grid_height, (num_boxes_x, num_boxes_y)
    )

    if self.merge_similar_heights and not self.holes:
      box_list_, box_color_ = self._create_merged_boxes(
        spec,
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
      boxes_list.extend(
        self._create_individual_boxes(
          spec,
          height_map,
          num_boxes_x,
          num_boxes_y,
          terrain_height,
          border_width,
        )
      )

    # Platform
    platform_height = terrain_height + grid_height
    platform_center_z = -terrain_height / 2 + grid_height / 2
    half_platform = self.platform_width / 2

    box = spec.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(half_platform, half_platform, platform_height / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, platform_center_z),
    )
    boxes_list.append(box)
    platform_rgba = platform_from_base(base_green)
    box_colors.append(platform_rgba)

    origin = np.array([self.size[0] / 2, self.size[1] / 2, grid_height])

    return origin, boxes_list, box_colors

  def _create_merged_boxes(
    self,
    spec,
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

    base_green = (0.25, 0.80, 0.45)

    for i in range(num_boxes_x):
      for j in range(num_boxes_y):
        if visited[i, j]:
          continue

        # Find rectangular region with similar height
        height = quantized_heights[i, j]

        normalized_height = (height + grid_height) / (2 * grid_height)
        t = float(np.clip(normalized_height, 0.0, 1.0))
        rgba = brand_ramp(base_green, t)

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

        box = spec.worldbody.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(width_x / 2, width_y / 2, box_height / 2),
          pos=(box_center_x, box_center_y, box_center_z),
        )
        boxes.append(box)
        box_colors.append(rgba)

    return boxes, box_colors

  def _create_individual_boxes(
    self,
    spec,
    height_map,
    num_boxes_x,
    num_boxes_y,
    terrain_height,
    border_width,
  ):
    """Original approach with individual boxes."""
    boxes = []
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

        box = spec.worldbody.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(half_grid, half_grid, box_height / 2),
          pos=(box_center_x, box_center_y, box_center_z),
        )
        boxes.append(box)

    return boxes


@dataclass(kw_only=True)
class BoxRepeatedTerrainCfg(SubTerrainCfg):
  @dataclass
  class ObjectCfg:
    num_objects: int
    height: float
    size: tuple[float, float]
    max_yx_angle: float = 0.0
    degrees: bool = True

  object_params_start: ObjectCfg
  object_params_end: ObjectCfg
  abs_height_noise: tuple[float, float] = (0.0, 0.0)
  rel_height_noise: tuple[float, float] = (1.0, 1.0)
  platform_width: float = 1.0
  platform_height: float = -1.0

  def function(self, difficulty: float, spec: mujoco.MjSpec):
    cp_0 = self.object_params_start
    cp_1 = self.object_params_end

    num_objects = cp_0.num_objects + int(
      difficulty * (cp_1.num_objects - cp_0.num_objects)
    )
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    platform_height = self.platform_height if self.platform_height >= 0.0 else height

    object_kwargs = {
      "length": cp_0.size[0] + difficulty * (cp_1.size[0] - cp_0.size[0]),
      "width": cp_0.size[1] + difficulty * (cp_1.size[1] - cp_0.size[1]),
      "max_yx_angle": cp_0.max_yx_angle
      + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
      "degrees": cp_0.degrees,
    }

    platform_clearance = 0.1

    boxes = []

    def make_box(
      spec: mujoco.MjSpec,
      length: float,
      width: float,
      height: float,
      center: tuple[float, float, float],
      max_yx_angle: float = 0.0,
      degrees: bool = True,
    ):
      from scipy.spatial.transform import Rotation as R

      euler_zyx = np.array(
        [
          np.random.uniform(0, 2 * np.pi),  # Z: full rotation (0 to 2π)
          np.random.uniform(-1, 1),  # Y: normalized (-1 to 1)
          np.random.uniform(-1, 1),  # X: normalized (-1 to 1)
        ]
      )
      if degrees:
        max_yx_angle = max_yx_angle / 180.0
      euler_zyx[1:] *= max_yx_angle
      dims = (length, width, height)
      quat = R.from_euler("zyx", euler_zyx).as_quat(scalar_first=True)
      box = spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0),
        pos=center,
        quat=quat,
      )
      return box

    origin = np.asarray((0.5 * self.size[0], 0.5 * self.size[1], 0.5 * platform_height))
    platform_corners = np.asarray(
      [
        [origin[0] - self.platform_width / 2, origin[1] - self.platform_width / 2],
        [origin[0] + self.platform_width / 2, origin[1] + self.platform_width / 2],
      ]
    )
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance

    object_centers = np.zeros((num_objects, 3))
    # use a mask to track invalid objects that still require sampling
    mask_objects_left = np.ones((num_objects,), dtype=bool)

    while np.any(mask_objects_left):
      num_objects_left = mask_objects_left.sum()
      object_centers[mask_objects_left, 0] = np.random.uniform(
        0, self.size[0], num_objects_left
      )
      object_centers[mask_objects_left, 1] = np.random.uniform(
        0, self.size[1], num_objects_left
      )
      # filter out the centers that are on the platform
      is_within_platform_x = np.logical_and(
        object_centers[mask_objects_left, 0] >= platform_corners[0, 0],
        object_centers[mask_objects_left, 0] <= platform_corners[1, 0],
      )
      is_within_platform_y = np.logical_and(
        object_centers[mask_objects_left, 1] >= platform_corners[0, 1],
        object_centers[mask_objects_left, 1] <= platform_corners[1, 1],
      )
      # update the mask to track the validity of the objects sampled in this iteration
      mask_objects_left[mask_objects_left] = np.logical_and(
        is_within_platform_x, is_within_platform_y
      )

    for index in range(len(object_centers)):
      abs_height_noise = np.random.uniform(
        self.abs_height_noise[0], self.abs_height_noise[1]
      )
      rel_height_noise = np.random.uniform(
        self.rel_height_noise[0], self.rel_height_noise[1]
      )
      ob_height = height * rel_height_noise + abs_height_noise
      if ob_height > 0.0:
        boxes.append(
          make_box(
            spec, center=object_centers[index], height=ob_height, **object_kwargs
          )
        )

    # generate a ground plane for the terrain
    box = spec.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.size[1] / 2.0, 1e-3),
      pos=(self.size[0] / 2.0, self.size[1] / 2.0, 0.0),
    )
    boxes.append(box)

    # Generate a platform in the middle.
    dim = (self.platform_width, self.platform_width, 0.5 * platform_height)
    pos = (0.5 * self.size[0], 0.5 * self.size[1], 0.25 * platform_height)
    box = spec.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(dim[0] / 2.0, dim[1] / 2.0, dim[2] / 2.0),
      pos=pos,
    )
    boxes.append(box)

    return origin, boxes
