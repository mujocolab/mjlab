"""Convert MuJoCo mesh data to trimesh format with texture support."""

import mujoco
import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj
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

  # Get the mesh ID for this geometry.
  mesh_id = mj_model.geom_dataid[geom_idx]

  # Get mesh data ranges from MuJoCo.
  vert_start = int(mj_model.mesh_vertadr[mesh_id])
  vert_count = int(mj_model.mesh_vertnum[mesh_id])
  face_start = int(mj_model.mesh_faceadr[mesh_id])
  face_count = int(mj_model.mesh_facenum[mesh_id])

  # Extract vertices and faces.
  # mesh_vert shape: (total_verts_in_model, 3)
  # We extract our mesh's vertices.
  vertices = mj_model.mesh_vert[
    vert_start : vert_start + vert_count
  ]  # Shape: (vert_count, 3)
  assert vertices.shape == (
    vert_count,
    3,
  ), f"Expected vertices shape ({vert_count}, 3), got {vertices.shape}"

  # mesh_face shape: (total_faces_in_model, 3)
  # Each face has 3 vertex indices.
  faces = mj_model.mesh_face[
    face_start : face_start + face_count
  ]  # Shape: (face_count, 3)
  assert faces.shape == (
    face_count,
    3,
  ), f"Expected faces shape ({face_count}, 3), got {faces.shape}"

  # Check if this mesh has texture coordinates.
  texcoord_adr = mj_model.mesh_texcoordadr[mesh_id]
  texcoord_num = mj_model.mesh_texcoordnum[mesh_id]

  if texcoord_num > 0:
    # This mesh has UV coordinates.
    if verbose:
      print(f"Mesh has {texcoord_num} texture coordinates")

    # Extract texture coordinates.
    # mesh_texcoord is a 2D array with shape (nmeshtexcoord, 2).
    texcoords = mj_model.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_num]
    assert texcoords.shape == (
      texcoord_num,
      2,
    ), f"Expected texcoords shape ({texcoord_num}, 2), got {texcoords.shape}"

    # Get per-face texture coordinate indices.
    # For each face vertex, this tells us which texcoord to use.
    # mesh_facetexcoord is a 2D array with shape (nmeshface, 3).
    face_texcoord_idx = mj_model.mesh_facetexcoord[face_start : face_start + face_count]
    assert face_texcoord_idx.shape == (face_count, 3), (
      f"Expected face_texcoord_idx shape ({face_count}, 3), got {face_texcoord_idx.shape}"
    )

    # Since the same vertex can have different UVs in different faces,
    # we need to duplicate vertices. Each face will get its own 3 vertices.

    # Duplicate vertices for each face reference.
    # faces.flatten() gives us vertex indices in order: [v0_f0, v1_f0, v2_f0, v0_f1, v1_f1, v2_f1, ...]
    new_vertices = vertices[faces.flatten()]  # Shape: (face_count * 3, 3)
    assert new_vertices.shape == (
      face_count * 3,
      3,
    ), f"Expected new_vertices shape ({face_count * 3}, 3), got {new_vertices.shape}"

    # Get UV coordinates for each duplicated vertex.
    # face_texcoord_idx.flatten() gives us texcoord indices in the same order.
    new_uvs = texcoords[face_texcoord_idx.flatten()]  # Shape: (face_count * 3, 2)
    assert new_uvs.shape == (
      face_count * 3,
      2,
    ), f"Expected new_uvs shape ({face_count * 3}, 2), got {new_uvs.shape}"

    # Create new faces - now just sequential since vertices are duplicated.
    # [[0, 1, 2], [3, 4, 5], [6, 7, 8], ...]
    new_faces = np.arange(face_count * 3).reshape(-1, 3)  # Shape: (face_count, 3)
    assert new_faces.shape == (
      face_count,
      3,
    ), f"Expected new_faces shape ({face_count}, 3), got {new_faces.shape}"

    # Create the mesh (process=False to preserve all vertices).
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    # Now handle material and texture.
    matid = mj_model.geom_matid[geom_idx]

    if matid >= 0 and matid < mj_model.nmat:
      # This geometry has a material.
      rgba = mj_model.mat_rgba[matid]  # Shape: (4,)
      # mat_texid is 2D (nmat x mjNTEXROLE), get the RGB/RGBA texture.
      # Try RGB first (index 1), then RGBA (index 8).
      texid = int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
      if texid < 0:
        texid = int(
          mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)]
        )

      if texid >= 0 and texid < mj_model.ntex:
        # This material has a texture.
        if verbose:
          print(f"Material has texture ID {texid}")

        # Extract texture data.
        tex_width = mj_model.tex_width[texid]
        tex_height = mj_model.tex_height[texid]
        tex_nchannel = mj_model.tex_nchannel[texid]
        tex_adr = mj_model.tex_adr[texid]

        # Calculate texture data size.
        tex_size = tex_width * tex_height * tex_nchannel

        # Extract raw texture data.
        tex_data = mj_model.tex_data[tex_adr : tex_adr + tex_size]
        assert tex_data.shape == (tex_size,), (
          f"Expected tex_data shape ({tex_size},), got {tex_data.shape}"
        )

        # Reshape texture data based on number of channels.
        # Note: MuJoCo uses OpenGL convention (origin at bottom-left)
        # but GLTF/GLB expects top-left origin, so we flip vertically.
        if tex_nchannel == 1:
          # Grayscale.
          tex_array = tex_data.reshape(tex_height, tex_width)
          # Flip vertically for GLTF convention.
          tex_array = np.flipud(tex_array)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="L")
        elif tex_nchannel == 3:
          # RGB.
          tex_array = tex_data.reshape(tex_height, tex_width, 3)
          # Flip vertically for GLTF convention.
          tex_array = np.flipud(tex_array)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="RGB")
        elif tex_nchannel == 4:
          # RGBA.
          tex_array = tex_data.reshape(tex_height, tex_width, 4)
          # Flip vertically for GLTF convention.
          tex_array = np.flipud(tex_array)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="RGBA")
        else:
          if verbose:
            print(f"Unsupported number of texture channels: {tex_nchannel}")
          image = None

        if image is not None:
          # Create material with texture.
          # Set PBR properties for proper rendering:
          # - metallicFactor=0.0: non-metallic (dielectric) material
          # - roughnessFactor=1.0: fully rough (diffuse) surface
          material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=rgba,
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
          )

          # Apply texture visual with UV coordinates.
          mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
          if verbose:
            print(f"Applied texture: {tex_width}x{tex_height}, {tex_nchannel} channels")
        else:
          # Just use material color - convert from [0,1] to [0,255].
          rgba_255 = (rgba * 255).astype(np.uint8)
          mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
          )
      else:
        # Material but no texture - use material color.
        if verbose:
          print(f"Material has no texture, using color: {rgba}")
        rgba_255 = (rgba * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
          vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
        )
    else:
      # No material - use default color based on collision/visual.
      is_collision = (
        mj_model.geom_contype[geom_idx] != 0 or mj_model.geom_conaffinity[geom_idx] != 0
      )
      if is_collision:
        color = np.array([204, 102, 102, 128], dtype=np.uint8)  # Red-ish for collision.
      else:
        color = np.array([31, 128, 230, 255], dtype=np.uint8)  # Blue-ish for visual.

      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(color, (len(new_vertices), 1))
      )
      if verbose:
        print(
          f"No material, using default {'collision' if is_collision else 'visual'} color"
        )

  else:
    # No texture coordinates - simpler case.
    if verbose:
      print("Mesh has no texture coordinates")

    # Create mesh with original vertices and faces (process=False to avoid vertex removal).
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Apply material color if available.
    matid = mj_model.geom_matid[geom_idx]

    if matid >= 0 and matid < mj_model.nmat:
      rgba = mj_model.mat_rgba[matid]
      rgba_255 = (rgba * 255).astype(np.uint8)
      # Use actual vertex count after mesh creation.
      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1))
      )
      if verbose:
        print(f"Applied material color: {rgba}")
    else:
      # Default color.
      is_collision = (
        mj_model.geom_contype[geom_idx] != 0 or mj_model.geom_conaffinity[geom_idx] != 0
      )
      if is_collision:
        color = np.array([204, 102, 102, 128], dtype=np.uint8)  # Red-ish for collision.
      else:
        color = np.array([31, 128, 230, 255], dtype=np.uint8)  # Blue-ish for visual.

      # Use actual vertex count after mesh creation.
      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(color, (len(mesh.vertices), 1))
      )
      if verbose:
        print(f"Using default {'collision' if is_collision else 'visual'} color")

  # Final sanity checks.
  assert mesh.vertices.shape[1] == 3, (
    f"Vertices should be Nx3, got {mesh.vertices.shape}"
  )
  assert mesh.faces.shape[1] == 3, f"Faces should be Nx3, got {mesh.faces.shape}"
  assert len(mesh.vertices) > 0, "Mesh has no vertices"
  assert len(mesh.faces) > 0, "Mesh has no faces"

  if verbose:
    print(f"Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

  return mesh


def create_primitive_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
  """Create a mesh for primitive geom types (sphere, box, capsule, cylinder, plane).

  Args:
    mj_model: MuJoCo model containing geom definition
    geom_id: Index of the geom to create mesh for

  Returns:
    Trimesh representation of the primitive geom
  """
  size = mj_model.geom_size[geom_id]
  geom_type = mj_model.geom_type[geom_id]
  rgba = mj_model.geom_rgba[geom_id].copy()

  material = trimesh.visual.material.PBRMaterial(  # type: ignore
    baseColorFactor=rgba,
    metallicFactor=0.0,
    roughnessFactor=1.0,
  )

  if geom_type == mjtGeom.mjGEOM_SPHERE:
    mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
  elif geom_type == mjtGeom.mjGEOM_BOX:
    mesh = trimesh.creation.box(extents=2.0 * size)
  elif geom_type == mjtGeom.mjGEOM_CAPSULE:
    mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_CYLINDER:
    mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_PLANE:
    mesh = trimesh.creation.box((20, 20, 0.01))
  else:
    raise ValueError(f"Unsupported primitive geom type: {geom_type}")

  mesh.visual = trimesh.visual.TextureVisuals(material=material)  # type: ignore
  return mesh


def merge_geoms(mj_model: mujoco.MjModel, geom_ids: list[int]) -> trimesh.Trimesh:
  """Merge multiple geoms into a single trimesh.

  Args:
    mj_model: MuJoCo model containing geom definitions
    geom_ids: List of geom indices to merge

  Returns:
    Single merged trimesh with all geoms transformed to their local poses
  """
  meshes = []
  for geom_id in geom_ids:
    geom_type = mj_model.geom_type[geom_id]

    if geom_type == mjtGeom.mjGEOM_MESH:
      mesh = mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
    else:
      mesh = create_primitive_mesh(mj_model, geom_id)

    pos = mj_model.geom_pos[geom_id]
    quat = mj_model.geom_quat[geom_id]
    transform = np.eye(4)
    transform[:3, :3] = vtf.SO3(quat).as_matrix()
    transform[:3, 3] = pos
    mesh.apply_transform(transform)
    meshes.append(mesh)

  if len(meshes) == 1:
    return meshes[0]
  return trimesh.util.concatenate(meshes)


def rotation_quat_from_vectors(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
  """Compute quaternion (wxyz format) that rotates from_vec to to_vec.

  Args:
    from_vec: Source vector (3D)
    to_vec: Target vector (3D)

  Returns:
    Quaternion in wxyz format that rotates from_vec to to_vec.
  """
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)

  if np.allclose(from_vec, to_vec):
    return np.array([1.0, 0.0, 0.0, 0.0])

  if np.allclose(from_vec, -to_vec):
    # 180 degree rotation - pick arbitrary perpendicular axis.
    perp = np.array([1.0, 0.0, 0.0])
    if abs(from_vec[0]) > 0.9:
      perp = np.array([0.0, 1.0, 0.0])
    axis = np.cross(from_vec, perp)
    axis = axis / np.linalg.norm(axis)
    return np.array([0.0, axis[0], axis[1], axis[2]])  # wxyz for 180 deg.

  # Standard quaternion from two vectors.
  cross = np.cross(from_vec, to_vec)
  dot = np.dot(from_vec, to_vec)
  w = 1.0 + dot
  quat = np.array([w, cross[0], cross[1], cross[2]])
  quat = quat / np.linalg.norm(quat)
  return quat


def rotation_matrix_from_vectors(
  from_vec: np.ndarray, to_vec: np.ndarray
) -> np.ndarray:
  """Create rotation matrix that rotates from_vec to to_vec using Rodrigues formula.

  Args:
    from_vec: Source vector (3D)
    to_vec: Target vector (3D)

  Returns:
    3x3 rotation matrix that rotates from_vec to to_vec.
  """
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)

  if np.allclose(from_vec, to_vec):
    return np.eye(3)

  if np.allclose(from_vec, -to_vec):
    return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

  # Rodrigues rotation formula.
  v = np.cross(from_vec, to_vec)
  s = np.linalg.norm(v)
  c = np.dot(from_vec, to_vec)
  vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def is_fixed_body(mj_model: mujoco.MjModel, body_id: int) -> bool:
  """Check if a body is fixed (welded to world)."""
  if mj_model.body_mocapid[body_id] >= 0:
    return False
  return mj_model.body_weldid[body_id] == 0


def get_body_name(mj_model: mujoco.MjModel, body_id: int) -> str:
  """Get body name with fallback to ID-based name.

  Args:
    mj_model: MuJoCo model
    body_id: Body index

  Returns:
    Body name or "body_{body_id}" if name not found.
  """
  body_name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
  if not body_name:
    body_name = f"body_{body_id}"
  return body_name
