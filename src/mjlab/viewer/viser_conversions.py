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

"""Convert MuJoCo mesh data to trimesh format with texture support."""

import mujoco
import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
from PIL import Image


def mujoco_mesh_to_trimesh(
  mj_model: mujoco.MjModel, geom_idx: int, verbose: bool = False
) -> trimesh.Trimesh:
  """Convert a MuJoCo mesh geometry to a trimesh with textures if available.

  Args:
      mj_model: MuJoCo model object
      geom_idx: Index of the geometry in the model
      verbose: If True, print debug information during conversion

  Returns:
      A trimesh object with texture/material applied if available
  """

  # Get the mesh ID for this geometry
  mesh_id = mj_model.geom_dataid[geom_idx]

  # Get mesh data ranges from MuJoCo
  vert_start = int(mj_model.mesh_vertadr[mesh_id])
  vert_count = int(mj_model.mesh_vertnum[mesh_id])
  face_start = int(mj_model.mesh_faceadr[mesh_id])
  face_count = int(mj_model.mesh_facenum[mesh_id])

  # Extract vertices and faces
  # mesh_vert shape: (total_verts_in_model, 3)
  # We extract our mesh's vertices
  vertices = mj_model.mesh_vert[
    vert_start : vert_start + vert_count
  ]  # Shape: (vert_count, 3)
  assert vertices.shape == (
    vert_count,
    3,
  ), f"Expected vertices shape ({vert_count}, 3), got {vertices.shape}"

  # mesh_face shape: (total_faces_in_model, 3)
  # Each face has 3 vertex indices
  faces = mj_model.mesh_face[
    face_start : face_start + face_count
  ]  # Shape: (face_count, 3)
  assert faces.shape == (
    face_count,
    3,
  ), f"Expected faces shape ({face_count}, 3), got {faces.shape}"

  # Check if this mesh has texture coordinates
  texcoord_adr = mj_model.mesh_texcoordadr[mesh_id]
  texcoord_num = mj_model.mesh_texcoordnum[mesh_id]

  if texcoord_num > 0:
    # This mesh has UV coordinates
    if verbose:
      print(f"Mesh has {texcoord_num} texture coordinates")

    # Extract texture coordinates
    # mesh_texcoord is a flat array of (u, v) pairs
    texcoords_flat = mj_model.mesh_texcoord[
      texcoord_adr : texcoord_adr + texcoord_num * 2
    ]
    assert texcoords_flat.shape == (texcoord_num * 2,), (
      f"Expected texcoords shape ({texcoord_num * 2},), got {texcoords_flat.shape}"
    )

    # Reshape to (N, 2) for easier indexing
    texcoords = texcoords_flat.reshape(-1, 2)  # Shape: (texcoord_num, 2)
    assert texcoords.shape == (
      texcoord_num,
      2,
    ), f"Expected texcoords shape ({texcoord_num}, 2), got {texcoords.shape}"

    # Get per-face texture coordinate indices
    # For each face vertex, this tells us which texcoord to use
    face_texcoord_idx = mj_model.mesh_facetexcoord[
      face_start * 3 : (face_start + face_count) * 3
    ]
    assert face_texcoord_idx.shape == (face_count * 3,), (
      f"Expected face_texcoord_idx shape ({face_count * 3},), got {face_texcoord_idx.shape}"
    )

    # Reshape to match faces shape
    face_texcoord_idx = face_texcoord_idx.reshape(
      face_count, 3
    )  # Shape: (face_count, 3)
    assert face_texcoord_idx.shape == (face_count, 3), (
      f"Expected face_texcoord_idx shape ({face_count}, 3), got {face_texcoord_idx.shape}"
    )

    # Since the same vertex can have different UVs in different faces,
    # we need to duplicate vertices. Each face will get its own 3 vertices.

    # Duplicate vertices for each face reference
    # faces.flatten() gives us vertex indices in order: [v0_f0, v1_f0, v2_f0, v0_f1, v1_f1, v2_f1, ...]
    new_vertices = vertices[faces.flatten()]  # Shape: (face_count * 3, 3)
    assert new_vertices.shape == (
      face_count * 3,
      3,
    ), f"Expected new_vertices shape ({face_count * 3}, 3), got {new_vertices.shape}"

    # Get UV coordinates for each duplicated vertex
    # face_texcoord_idx.flatten() gives us texcoord indices in the same order
    new_uvs = texcoords[face_texcoord_idx.flatten()]  # Shape: (face_count * 3, 2)
    assert new_uvs.shape == (
      face_count * 3,
      2,
    ), f"Expected new_uvs shape ({face_count * 3}, 2), got {new_uvs.shape}"

    # Create new faces - now just sequential since vertices are duplicated
    # [[0, 1, 2], [3, 4, 5], [6, 7, 8], ...]
    new_faces = np.arange(face_count * 3).reshape(-1, 3)  # Shape: (face_count, 3)
    assert new_faces.shape == (
      face_count,
      3,
    ), f"Expected new_faces shape ({face_count}, 3), got {new_faces.shape}"

    # Create the mesh (process=False to preserve all vertices)
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    # Now handle material and texture
    matid = mj_model.geom_matid[geom_idx]

    if matid >= 0 and matid < mj_model.nmat:
      # This geometry has a material
      rgba = mj_model.mat_rgba[matid]  # Shape: (4,)
      texid = mj_model.mat_texid[matid]

      if texid >= 0 and texid < mj_model.ntex:
        # This material has a texture
        if verbose:
          print(f"Material has texture ID {texid}")

        # Extract texture data
        tex_width = mj_model.tex_width[texid]
        tex_height = mj_model.tex_height[texid]
        tex_nchannel = mj_model.tex_nchannel[texid]
        tex_adr = mj_model.tex_adr[texid]

        # Calculate texture data size
        tex_size = tex_width * tex_height * tex_nchannel

        # Extract raw texture data
        tex_data = mj_model.tex_data[tex_adr : tex_adr + tex_size]
        assert tex_data.shape == (tex_size,), (
          f"Expected tex_data shape ({tex_size},), got {tex_data.shape}"
        )

        # Reshape texture data based on number of channels
        if tex_nchannel == 1:
          # Grayscale
          tex_array = tex_data.reshape(tex_height, tex_width)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="L")
        elif tex_nchannel == 3:
          # RGB
          tex_array = tex_data.reshape(tex_height, tex_width, 3)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="RGB")
        elif tex_nchannel == 4:
          # RGBA
          tex_array = tex_data.reshape(tex_height, tex_width, 4)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="RGBA")
        else:
          if verbose:
            print(f"Unsupported number of texture channels: {tex_nchannel}")
          image = None

        if image is not None:
          # Create material with texture
          material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=rgba, baseColorTexture=image
          )

          # Apply texture visual with UV coordinates
          mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
          if verbose:
            print(f"Applied texture: {tex_width}x{tex_height}, {tex_nchannel} channels")
        else:
          # Just use material color - convert from [0,1] to [0,255]
          rgba_255 = (rgba * 255).astype(np.uint8)
          mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
          )
      else:
        # Material but no texture - use material color
        if verbose:
          print(f"Material has no texture, using color: {rgba}")
        rgba_255 = (rgba * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
          vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
        )
    else:
      # No material - use default color based on collision/visual
      is_collision = (
        mj_model.geom_contype[geom_idx] != 0 or mj_model.geom_conaffinity[geom_idx] != 0
      )
      if is_collision:
        color = np.array([204, 102, 102, 128], dtype=np.uint8)  # Red-ish for collision
      else:
        color = np.array([31, 128, 230, 255], dtype=np.uint8)  # Blue-ish for visual

      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(color, (len(new_vertices), 1))
      )
      if verbose:
        print(
          f"No material, using default {'collision' if is_collision else 'visual'} color"
        )

  else:
    # No texture coordinates - simpler case
    if verbose:
      print("Mesh has no texture coordinates")

    # Create mesh with original vertices and faces (process=False to avoid vertex removal)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Apply material color if available
    matid = mj_model.geom_matid[geom_idx]

    if matid >= 0 and matid < mj_model.nmat:
      rgba = mj_model.mat_rgba[matid]
      rgba_255 = (rgba * 255).astype(np.uint8)
      # Use actual vertex count after mesh creation
      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1))
      )
      if verbose:
        print(f"Applied material color: {rgba}")
    else:
      # Default color
      is_collision = (
        mj_model.geom_contype[geom_idx] != 0 or mj_model.geom_conaffinity[geom_idx] != 0
      )
      if is_collision:
        color = np.array([204, 102, 102, 128], dtype=np.uint8)  # Red-ish for collision
      else:
        color = np.array([31, 128, 230, 255], dtype=np.uint8)  # Blue-ish for visual

      # Use actual vertex count after mesh creation
      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(color, (len(mesh.vertices), 1))
      )
      if verbose:
        print(f"Using default {'collision' if is_collision else 'visual'} color")

  # Final sanity checks
  assert mesh.vertices.shape[1] == 3, (
    f"Vertices should be Nx3, got {mesh.vertices.shape}"
  )
  assert mesh.faces.shape[1] == 3, f"Faces should be Nx3, got {mesh.faces.shape}"
  assert len(mesh.vertices) > 0, "Mesh has no vertices"
  assert len(mesh.faces) > 0, "Mesh has no faces"

  if verbose:
    print(f"Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

  return mesh
