from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

from mjlab.third_party.isaaclab.isaaclab.terrains.trimesh.utils import make_border

if TYPE_CHECKING:
  from mjlab.terrains import mesh_terrains_cfg


def pyramid_stairs_terrain(
  difficulty: float, cfg: mesh_terrains_cfg.MeshPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
  """Generate a terrain with a pyramid stair pattern.

  The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

  If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
  :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
  no border will be added.

  .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain.jpg
     :width: 45%

  .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain_with_holes.jpg
     :width: 45%

  Args:
      difficulty: The difficulty of the terrain. This is a value between 0 and 1.
      cfg: The configuration for the terrain.

  Returns:
      A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
  """
  # resolve the terrain configuration
  step_height = cfg.step_height_range[0] + difficulty * (
    cfg.step_height_range[1] - cfg.step_height_range[0]
  )

  # compute number of steps in x and y direction
  num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (
    2 * cfg.step_width
  ) + 1
  num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (
    2 * cfg.step_width
  ) + 1
  # we take the minimum number of steps in x and y direction
  num_steps = int(min(num_steps_x, num_steps_y))

  # initialize list of meshes
  meshes_list = list()

  # generate the border if needed
  if cfg.border_width > 0.0 and not cfg.holes:
    # obtain a list of meshes for the border
    border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
    border_inner_size = (
      cfg.size[0] - 2 * cfg.border_width,
      cfg.size[1] - 2 * cfg.border_width,
    )
    make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
    # add the border meshes to the list of meshes
    meshes_list += make_borders

  # generate the terrain
  # -- compute the position of the center of the terrain
  terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
  terrain_size = (
    cfg.size[0] - 2 * cfg.border_width,
    cfg.size[1] - 2 * cfg.border_width,
  )
  # -- generate the stair pattern
  for k in range(num_steps):
    # check if we need to add holes around the steps
    if cfg.holes:
      box_size = (cfg.platform_width, cfg.platform_width)
    else:
      box_size = (
        terrain_size[0] - 2 * k * cfg.step_width,
        terrain_size[1] - 2 * k * cfg.step_width,
      )
    # compute the quantities of the box
    # -- location
    box_z = terrain_center[2] + k * step_height / 2.0
    box_offset = (k + 0.5) * cfg.step_width
    # -- dimensions
    box_height = (k + 2) * step_height
    # generate the boxes
    # top/bottom
    box_dims = (box_size[0], cfg.step_width, box_height)
    # -- top
    box_pos = (
      terrain_center[0],
      terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
      box_z,
    )
    box_top = trimesh.creation.box(
      box_dims, trimesh.transformations.translation_matrix(box_pos)
    )
    # -- bottom
    box_pos = (
      terrain_center[0],
      terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
      box_z,
    )
    box_bottom = trimesh.creation.box(
      box_dims, trimesh.transformations.translation_matrix(box_pos)
    )
    # right/left
    if cfg.holes:
      box_dims = (cfg.step_width, box_size[1], box_height)
    else:
      box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
    # -- right
    box_pos = (
      terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
      terrain_center[1],
      box_z,
    )
    box_right = trimesh.creation.box(
      box_dims, trimesh.transformations.translation_matrix(box_pos)
    )
    # -- left
    box_pos = (
      terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
      terrain_center[1],
      box_z,
    )
    box_left = trimesh.creation.box(
      box_dims, trimesh.transformations.translation_matrix(box_pos)
    )
    # add the boxes to the list of meshes
    meshes_list += [box_top, box_bottom, box_right, box_left]

  # generate final box for the middle of the terrain
  box_dims = (
    terrain_size[0] - 2 * num_steps * cfg.step_width,
    terrain_size[1] - 2 * num_steps * cfg.step_width,
    (num_steps + 2) * step_height,
  )
  box_pos = (
    terrain_center[0],
    terrain_center[1],
    terrain_center[2] + num_steps * step_height / 2,
  )
  box_middle = trimesh.creation.box(
    box_dims, trimesh.transformations.translation_matrix(box_pos)
  )
  meshes_list.append(box_middle)
  # origin of the terrain
  origin = np.array(
    [terrain_center[0], terrain_center[1], (num_steps + 1) * step_height]
  )

  return meshes_list, origin


def inverted_pyramid_stairs_terrain(
  difficulty: float, cfg: mesh_terrains_cfg.MeshInvertedPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
  """Generate a terrain with a inverted pyramid stair pattern.

  The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

  If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
  :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
  no border will be added.

  .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain.jpg
     :width: 45%

  .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain_with_holes.jpg
     :width: 45%

  Args:
      difficulty: The difficulty of the terrain. This is a value between 0 and 1.
      cfg: The configuration for the terrain.

  Returns:
      A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
  """
  # resolve the terrain configuration
  step_height = cfg.step_height_range[0] + difficulty * (
    cfg.step_height_range[1] - cfg.step_height_range[0]
  )

  # compute number of steps in x and y direction
  num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (
    2 * cfg.step_width
  ) + 1
  num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (
    2 * cfg.step_width
  ) + 1
  # we take the minimum number of steps in x and y direction
  num_steps = int(min(num_steps_x, num_steps_y))
  # total height of the terrain
  total_height = (num_steps + 1) * step_height

  # initialize list of meshes
  meshes_list = list()

  # generate the border if needed
  if cfg.border_width > 0.0 and not cfg.holes:
    # obtain a list of meshes for the border
    border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
    border_inner_size = (
      cfg.size[0] - 2 * cfg.border_width,
      cfg.size[1] - 2 * cfg.border_width,
    )
    make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
    # add the border meshes to the list of meshes
    meshes_list += make_borders
  # generate the terrain
  # -- compute the position of the center of the terrain
  terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
  terrain_size = (
    cfg.size[0] - 2 * cfg.border_width,
    cfg.size[1] - 2 * cfg.border_width,
  )
  # -- generate the stair pattern
  for k in range(num_steps):
    # check if we need to add holes around the steps
    if cfg.holes:
      box_size = (cfg.platform_width, cfg.platform_width)
    else:
      box_size = (
        terrain_size[0] - 2 * k * cfg.step_width,
        terrain_size[1] - 2 * k * cfg.step_width,
      )
    # compute the quantities of the box
    # -- location
    box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
    box_offset = (k + 0.5) * cfg.step_width
    # -- dimensions
    box_height = total_height - (k + 1) * step_height
    # generate the boxes
    # top/bottom
    box_dims = (box_size[0], cfg.step_width, box_height)
    # -- top
    box_pos = (
      terrain_center[0],
      terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
      box_z,
    )
    box_top = trimesh.creation.box(
      box_dims, trimesh.transformations.translation_matrix(box_pos)
    )
    # -- bottom
    box_pos = (
      terrain_center[0],
      terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
      box_z,
    )
    box_bottom = trimesh.creation.box(
      box_dims, trimesh.transformations.translation_matrix(box_pos)
    )
    # right/left
    if cfg.holes:
      box_dims = (cfg.step_width, box_size[1], box_height)
    else:
      box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
    # -- right
    box_pos = (
      terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
      terrain_center[1],
      box_z,
    )
    box_right = trimesh.creation.box(
      box_dims, trimesh.transformations.translation_matrix(box_pos)
    )
    # -- left
    box_pos = (
      terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
      terrain_center[1],
      box_z,
    )
    box_left = trimesh.creation.box(
      box_dims, trimesh.transformations.translation_matrix(box_pos)
    )
    # add the boxes to the list of meshes
    meshes_list += [box_top, box_bottom, box_right, box_left]
  # generate final box for the middle of the terrain
  box_dims = (
    terrain_size[0] - 2 * num_steps * cfg.step_width,
    terrain_size[1] - 2 * num_steps * cfg.step_width,
    step_height,
  )
  box_pos = (
    terrain_center[0],
    terrain_center[1],
    terrain_center[2] - total_height - step_height / 2,
  )
  box_middle = trimesh.creation.box(
    box_dims, trimesh.transformations.translation_matrix(box_pos)
  )
  meshes_list.append(box_middle)
  # origin of the terrain
  origin = np.array(
    [terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height]
  )

  return meshes_list, origin


def random_grid_terrain(
  difficulty: float, cfg: mesh_terrains_cfg.MeshRandomGridTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
  """Generate a terrain with cells of random heights and fixed width.

  The terrain is generated in the x-y plane and has a height of 1.0. It is then divided into a grid of the
  specified size :obj:`cfg.grid_width`. Each grid cell is then randomly shifted in the z-direction by a value uniformly
  sampled between :obj:`cfg.grid_height_range`. At the center of the terrain, a platform of the specified width
  :obj:`cfg.platform_width` is generated.

  If :obj:`cfg.holes` is True, the terrain will have randomized grid cells only along the plane extending
  from the platform (like a plus sign). The remaining area remains empty and no border will be added.

  .. image:: ../../_static/terrains/trimesh/random_grid_terrain.jpg
     :width: 45%

  .. image:: ../../_static/terrains/trimesh/random_grid_terrain_with_holes.jpg
     :width: 45%

  Args:
      difficulty: The difficulty of the terrain. This is a value between 0 and 1.
      cfg: The configuration for the terrain.

  Returns:
      A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

  Raises:
      ValueError: If the terrain is not square. This method only supports square terrains.
      RuntimeError: If the grid width is large such that the border width is negative.
  """
  # check to ensure square terrain
  if cfg.size[0] != cfg.size[1]:
    raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")
  # resolve the terrain configuration
  grid_height = cfg.grid_height_range[0] + difficulty * (
    cfg.grid_height_range[1] - cfg.grid_height_range[0]
  )

  # initialize list of meshes
  meshes_list = list()
  # compute the number of boxes in each direction
  num_boxes_x = int(cfg.size[0] / cfg.grid_width)
  num_boxes_y = int(cfg.size[1] / cfg.grid_width)
  # constant parameters
  terrain_height = 1.0
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  # generate the border
  border_width = cfg.size[0] - min(num_boxes_x, num_boxes_y) * cfg.grid_width
  if border_width > 0:
    # compute parameters for the border
    border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    border_inner_size = (cfg.size[0] - border_width, cfg.size[1] - border_width)
    # create border meshes
    make_borders = make_border(
      cfg.size, border_inner_size, terrain_height, border_center
    )
    meshes_list += make_borders
  else:
    raise RuntimeError(
      "Border width must be greater than 0! Adjust the parameter 'cfg.grid_width'."
    )

  # create a template grid of terrain height
  grid_dim = [cfg.grid_width, cfg.grid_width, terrain_height]
  grid_position = [0.5 * cfg.grid_width, 0.5 * cfg.grid_width, -terrain_height / 2]
  template_box = trimesh.creation.box(
    grid_dim, trimesh.transformations.translation_matrix(grid_position)
  )
  # extract vertices and faces of the box to create a template
  template_vertices = template_box.vertices  # (8, 3)
  template_faces = template_box.faces

  # repeat the template box vertices to span the terrain (num_boxes_x * num_boxes_y, 8, 3)
  vertices = torch.tensor(template_vertices, device=device).repeat(
    num_boxes_x * num_boxes_y, 1, 1
  )
  # create a meshgrid to offset the vertices
  x = torch.arange(0, num_boxes_x, device=device)
  y = torch.arange(0, num_boxes_y, device=device)
  xx, yy = torch.meshgrid(x, y, indexing="ij")
  xx = xx.flatten().view(-1, 1)
  yy = yy.flatten().view(-1, 1)
  xx_yy = torch.cat((xx, yy), dim=1)
  # offset the vertices
  offsets = cfg.grid_width * xx_yy + border_width / 2
  vertices[:, :, :2] += offsets.unsqueeze(1)
  # mask the vertices to create holes, s.t. only grids along the x and y axis are present
  if cfg.holes:
    # -- x-axis
    mask_x = torch.logical_and(
      (vertices[:, :, 0] > (cfg.size[0] - border_width - cfg.platform_width) / 2).all(
        dim=1
      ),
      (vertices[:, :, 0] < (cfg.size[0] + border_width + cfg.platform_width) / 2).all(
        dim=1
      ),
    )
    vertices_x = vertices[mask_x]
    # -- y-axis
    mask_y = torch.logical_and(
      (vertices[:, :, 1] > (cfg.size[1] - border_width - cfg.platform_width) / 2).all(
        dim=1
      ),
      (vertices[:, :, 1] < (cfg.size[1] + border_width + cfg.platform_width) / 2).all(
        dim=1
      ),
    )
    vertices_y = vertices[mask_y]
    # -- combine these vertices
    vertices = torch.cat((vertices_x, vertices_y))
  # add noise to the vertices to have a random height over each grid cell
  num_boxes = len(vertices)
  # create noise for the z-axis
  h_noise = torch.zeros((num_boxes, 3), device=device)
  h_noise[:, 2].uniform_(-grid_height, grid_height)
  # reshape noise to match the vertices (num_boxes, 4, 3)
  # only the top vertices of the box are affected
  vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
  vertices_noise += h_noise.unsqueeze(1)
  # add height only to the top vertices of the box
  vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
  # move to numpy
  # vertices = vertices.reshape(-1, 3).cpu().numpy()
  vertices = vertices.cpu().numpy()

  # create faces for boxes (num_boxes, 12, 3). Each box has 6 faces, each face has 2 triangles.
  faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
  # face_offsets = (
  #   torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
  # )
  # faces += face_offsets.unsqueeze(2)
  # move to numpy
  # faces = faces.view(-1, 3).cpu().numpy()
  faces = faces.cpu().numpy()
  # convert to trimesh
  # grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
  for v, f in zip(vertices, faces, strict=True):
    grid_mesh = trimesh.Trimesh(vertices=v, faces=f)
    meshes_list.append(grid_mesh)

  # add a platform in the center of the terrain that is accessible from all sides
  dim = (cfg.platform_width, cfg.platform_width, terrain_height + grid_height)
  pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2 + grid_height / 2)
  box_platform = trimesh.creation.box(
    dim, trimesh.transformations.translation_matrix(pos)
  )
  meshes_list.append(box_platform)

  # specify the origin of the terrain
  origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], grid_height])

  return meshes_list, origin
