"""Viser viewer debug visualizer implementation."""

from __future__ import annotations

import mujoco
import numpy as np
import torch
import trimesh
import viser
import viser.transforms as vtf


class ViserDebugVisualizer:
  """Debug visualizer for Viser viewer.

  This implementation uses Viser's scene graph to add visualization primitives
  like arrows and batched meshes.
  """

  def __init__(
    self,
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    env_idx: int,
    env_origin: np.ndarray | None = None,
  ):
    """Initialize the Viser debug visualizer.

    Args:
      server: Viser server instance
      mj_model: MuJoCo model (not used for ghost rendering, kept for compatibility)
      env_idx: Index of the environment being visualized
      env_origin: World origin offset for this environment
    """
    self.server = server
    self.mj_model = mj_model
    self.env_idx = env_idx
    self.env_origin = env_origin if env_origin is not None else np.zeros(3)

    # Track handles - we'll reuse them instead of recreating
    self._arrow_handles: list[viser.SceneNodeHandle] = []
    self._ghost_handles: dict[int, viser.SceneNodeHandle] = {}
    self._arrow_counter = 0

    # Cache ghost meshes by model hash to handle deepcopy'd models
    self._ghost_meshes: dict[int, dict[int, trimesh.Trimesh]] = {}

    # Cache arrow mesh for reuse
    self._arrow_mesh_cache: trimesh.Trimesh | None = None

    # Reusable MjData for ghost rendering
    self._viz_data = mujoco.MjData(mj_model)

  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Add an arrow visualization using Viser's scene primitives."""
    if isinstance(start, torch.Tensor):
      start = start.cpu().numpy()
    if isinstance(end, torch.Tensor):
      end = end.cpu().numpy()

    start = start + self.env_origin
    end = end + self.env_origin

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
      return

    direction = direction / length

    if self._arrow_mesh_cache is None:
      shaft_length = 0.8
      head_length = 0.2
      head_width = 3.0

      shaft_mesh = trimesh.creation.cylinder(radius=1.0, height=shaft_length)
      shaft_mesh.apply_translation(np.array([0, 0, shaft_length / 2]))

      head_mesh = trimesh.creation.cone(radius=head_width, height=head_length)
      head_mesh.apply_translation(np.array([0, 0, shaft_length + head_length / 2]))

      self._arrow_mesh_cache = trimesh.util.concatenate([shaft_mesh, head_mesh])

    assert self._arrow_mesh_cache is not None
    arrow_mesh = self._arrow_mesh_cache.copy()
    arrow_mesh.apply_scale([width, width, length])
    arrow_mesh.visual = trimesh.visual.TextureVisuals(  # type: ignore
      material=trimesh.visual.material.PBRMaterial(baseColorFactor=color)  # type: ignore
    )

    z_axis = np.array([0, 0, 1])
    rotation_quat = self._rotation_quat_from_vectors(z_axis, direction)

    arrow_name = f"/debug/env_{self.env_idx}/arrow_{self._arrow_counter}"
    arrow_handle = self.server.scene.add_mesh_trimesh(
      arrow_name,
      arrow_mesh,
      wxyz=rotation_quat,
      position=start,
    )
    self._arrow_handles.append(arrow_handle)
    self._arrow_counter += 1

  def add_ghost_mesh(
    self,
    qpos: np.ndarray | torch.Tensor,
    model: mujoco.MjModel,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost mesh by rendering the robot at a different pose.

    For Viser, we create meshes once and update their poses for efficiency.

    Args:
      qpos: Joint positions for the ghost pose
      model: MuJoCo model with pre-configured appearance (geom_rgba for colors)
      alpha: Transparency override
      label: Optional label for this ghost
    """
    if isinstance(qpos, torch.Tensor):
      qpos = qpos.cpu().numpy()

    # Use model hash to support models with same structure but different colors
    model_hash = hash((model.ngeom, model.nbody, model.nq))

    self._viz_data.qpos[:] = qpos
    mujoco.mj_forward(model, self._viz_data)

    # Group geoms by body
    body_geoms: dict[int, list[int]] = {}
    for i in range(model.ngeom):
      body_id = model.geom_bodyid[i]
      is_collision = model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
      if is_collision:
        continue

      if model.body_dofnum[body_id] == 0 and model.body_parentid[body_id] == 0:
        continue

      if body_id not in body_geoms:
        body_geoms[body_id] = []
      body_geoms[body_id].append(i)

    # Update or create mesh for each body
    for body_id, geom_indices in body_geoms.items():
      body_pos = self._viz_data.xpos[body_id] + self.env_origin
      body_quat = self._mat_to_quat(self._viz_data.xmat[body_id].reshape(3, 3))

      # Check if we already have a handle for this body
      if body_id in self._ghost_handles:
        handle = self._ghost_handles[body_id]
        handle.wxyz = body_quat
        handle.position = body_pos
      else:
        # Create mesh if not cached
        if model_hash not in self._ghost_meshes:
          self._ghost_meshes[model_hash] = {}

        if body_id not in self._ghost_meshes[model_hash]:
          meshes = []
          for geom_id in geom_indices:
            mesh = self._create_geom_mesh_from_model(model, geom_id)
            if mesh is not None:
              geom_pos = model.geom_pos[geom_id]
              geom_quat = model.geom_quat[geom_id]
              transform = np.eye(4)
              transform[:3, :3] = vtf.SO3(geom_quat).as_matrix()
              transform[:3, 3] = geom_pos
              mesh.apply_transform(transform)
              meshes.append(mesh)

          if not meshes:
            continue

          combined_mesh = (
            meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
          )

          # Apply alpha to geom colors
          rgba = model.geom_rgba[geom_indices[0]].copy()
          rgba[3] = alpha
          combined_mesh.visual = trimesh.visual.TextureVisuals(  # type: ignore
            material=trimesh.visual.material.PBRMaterial(baseColorFactor=rgba)  # type: ignore
          )

          self._ghost_meshes[model_hash][body_id] = combined_mesh
        else:
          combined_mesh = self._ghost_meshes[model_hash][body_id]

        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        handle_name = f"/debug/env_{self.env_idx}/ghost/body_{body_name or body_id}"
        handle = self.server.scene.add_mesh_trimesh(
          handle_name,
          combined_mesh,
          wxyz=body_quat,
          position=body_pos,
        )
        self._ghost_handles[body_id] = handle

  def _create_geom_mesh_from_model(
    self, mj_model: mujoco.MjModel, geom_id: int
  ) -> trimesh.Trimesh | None:
    """Create a trimesh from a MuJoCo geom using the specified model.

    Args:
      mj_model: MuJoCo model containing geom definition
      geom_id: Index of the geom to create mesh for

    Returns:
      Trimesh representation of the geom, or None if unsupported type
    """
    from mujoco import mjtGeom

    from mjlab.viewer.viser_conversions import mujoco_mesh_to_trimesh

    geom_type = mj_model.geom_type[geom_id]
    size = mj_model.geom_size[geom_id]
    rgba = mj_model.geom_rgba[geom_id].copy()

    material = trimesh.visual.material.PBRMaterial(  # type: ignore
      baseColorFactor=rgba,
      metallicFactor=0.5,
      roughnessFactor=0.5,
    )

    if geom_type == mjtGeom.mjGEOM_SPHERE:
      mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
    elif geom_type == mjtGeom.mjGEOM_BOX:
      dims = 2.0 * size
      mesh = trimesh.creation.box(extents=dims)
    elif geom_type == mjtGeom.mjGEOM_CAPSULE:
      mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    elif geom_type == mjtGeom.mjGEOM_CYLINDER:
      mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    elif geom_type == mjtGeom.mjGEOM_MESH:
      mesh = mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
    else:
      return None

    mesh.visual = trimesh.visual.TextureVisuals(material=material)  # type: ignore
    return mesh

  def clear(self) -> None:
    """Clear all debug visualizations.

    Removes arrows every frame. Ghost meshes are kept and pose-updated for efficiency
    within the same environment, but removed when switching environments.
    """
    for handle in self._arrow_handles:
      handle.remove()
    self._arrow_handles.clear()
    self._arrow_counter = 0

  def clear_all(self) -> None:
    """Clear all debug visualizations including ghosts.

    Called when switching to a different environment.
    """
    self.clear()
    for handle in self._ghost_handles.values():
      handle.remove()
    self._ghost_handles.clear()

  @staticmethod
  def _rotation_quat_from_vectors(
    from_vec: np.ndarray, to_vec: np.ndarray
  ) -> np.ndarray:
    """Compute quaternion (wxyz) that rotates from_vec to to_vec."""
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)

    if np.allclose(from_vec, to_vec):
      return np.array([1.0, 0.0, 0.0, 0.0])

    if np.allclose(from_vec, -to_vec):
      # 180 degree rotation - pick arbitrary perpendicular axis
      perp = np.array([1.0, 0.0, 0.0])
      if abs(from_vec[0]) > 0.9:
        perp = np.array([0.0, 1.0, 0.0])
      axis = np.cross(from_vec, perp)
      axis = axis / np.linalg.norm(axis)
      return np.array([0.0, axis[0], axis[1], axis[2]])  # wxyz for 180 deg

    # Standard quaternion from two vectors
    cross = np.cross(from_vec, to_vec)
    dot = np.dot(from_vec, to_vec)
    w = 1.0 + dot
    quat = np.array([w, cross[0], cross[1], cross[2]])
    quat = quat / np.linalg.norm(quat)
    return quat

  @staticmethod
  def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (wxyz)."""
    return vtf.SO3.from_matrix(mat).wxyz
