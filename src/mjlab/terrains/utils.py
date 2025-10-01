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

"""Utility functions for terrain generation.

References:
  IsaacLab terrain utilities:
  https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/trimesh/utils.py
"""

import mujoco


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
