"""Viser viewer debug visualizer implementation."""

from __future__ import annotations

import mujoco
import numpy as np
import torch
import trimesh
import trimesh.visual
import viser
import viser.transforms as vtf
from typing_extensions import override

from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.viser_conversions import get_body_name, rotation_quat_from_vectors


class ViserDebugVisualizer(DebugVisualizer):
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

    # Queued arrows for batched rendering
    self._queued_arrows: list[
      tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]
    ] = []

    # Batched arrow mesh handles
    self._arrow_shaft_handle: viser.BatchedMeshHandle | None = None
    self._arrow_head_handle: viser.BatchedMeshHandle | None = None

    # Ghost mesh handles
    self._ghost_handles: dict[int, viser.SceneNodeHandle] = {}

    # Cache ghost meshes by model hash to handle deepcopy'd models
    self._ghost_meshes: dict[int, dict[int, trimesh.Trimesh]] = {}

    # Cache arrow mesh components for batched rendering
    self._arrow_shaft_mesh: trimesh.Trimesh | None = None
    self._arrow_head_mesh: trimesh.Trimesh | None = None

    # Reusable MjData for ghost rendering
    self._viz_data = mujoco.MjData(mj_model)

  @override
  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Queue an arrow for batched rendering.

    Arrows are not rendered immediately but queued and rendered together
    in the next _synchronize() call for efficiency.
    """
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

    # Queue the arrow for batched rendering
    self._queued_arrows.append((start, end, color, width))

  @override
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

          self._ghost_meshes[model_hash][body_id] = combined_mesh
        else:
          combined_mesh = self._ghost_meshes[model_hash][body_id]

        body_name = get_body_name(model, body_id)
        handle_name = f"/debug/env_{self.env_idx}/ghost/body_{body_name}"

        # Extract color from geom (convert RGBA 0-1 to RGB 0-255)
        rgba = model.geom_rgba[geom_indices[0]].copy()
        color_uint8 = (rgba[:3] * 255).astype(np.uint8)

        handle = self.server.scene.add_mesh_simple(
          handle_name,
          combined_mesh.vertices,
          combined_mesh.faces,
          color=tuple(color_uint8),
          opacity=alpha,
          wxyz=body_quat,
          position=body_pos,
          cast_shadow=False,
          receive_shadow=False,
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

    from mjlab.viewer.viser_conversions import (
      create_primitive_mesh,
      mujoco_mesh_to_trimesh,
    )

    geom_type = mj_model.geom_type[geom_id]

    if geom_type == mjtGeom.mjGEOM_MESH:
      return mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
    else:
      return create_primitive_mesh(mj_model, geom_id)

  def _sync_arrows(self) -> None:
    """Render all queued arrows using batched meshes.

    This should be called by the main visualizer after all debug visualizations
    have been queued for the current frame.
    """
    if not self._queued_arrows:
      # Remove arrow meshes if no arrows to render
      if self._arrow_shaft_handle is not None:
        self._arrow_shaft_handle.remove()
        self._arrow_shaft_handle = None
      if self._arrow_head_handle is not None:
        self._arrow_head_handle.remove()
        self._arrow_head_handle = None
      return

    # Create arrow mesh components if needed (unit-sized base meshes)
    if self._arrow_shaft_mesh is None:
      # Unit cylinder: radius=1.0, height=1.0
      self._arrow_shaft_mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
      self._arrow_shaft_mesh.apply_translation(np.array([0, 0, 0.5]))  # Center at z=0.5

    if self._arrow_head_mesh is None:
      # Unit cone: radius=3.0, height=1.0 (base at z=0, tip at z=1.0 by default)
      head_width = 3.0
      self._arrow_head_mesh = trimesh.creation.cone(radius=head_width, height=1.0)
      # No translation needed - cone already has base at z=0

    # Prepare batched data
    num_arrows = len(self._queued_arrows)
    shaft_positions = np.zeros((num_arrows, 3), dtype=np.float32)
    shaft_wxyzs = np.zeros((num_arrows, 4), dtype=np.float32)
    shaft_scales = np.zeros((num_arrows, 3), dtype=np.float32)
    shaft_colors = np.zeros((num_arrows, 3), dtype=np.uint8)

    head_positions = np.zeros((num_arrows, 3), dtype=np.float32)
    head_wxyzs = np.zeros((num_arrows, 4), dtype=np.float32)
    head_scales = np.zeros((num_arrows, 3), dtype=np.float32)
    head_colors = np.zeros((num_arrows, 3), dtype=np.uint8)

    z_axis = np.array([0, 0, 1])
    shaft_length_ratio = 0.8
    head_length_ratio = 0.2

    for i, (start, end, color, width) in enumerate(self._queued_arrows):
      direction = end - start
      length = np.linalg.norm(direction)
      direction = direction / length

      rotation_quat = rotation_quat_from_vectors(z_axis, direction)

      # Shaft: scale width in XY, length in Z
      shaft_length = shaft_length_ratio * length
      shaft_positions[i] = start
      shaft_wxyzs[i] = rotation_quat
      shaft_scales[i] = [width, width, shaft_length]  # Per-axis scale
      shaft_colors[i] = (np.array(color[:3]) * 255).astype(
        np.uint8
      )  # Convert 0-1 to 0-255

      # Head: position at end of shaft
      # The cone has its base at z=0, so after scaling by head_length,
      # the base is still at z=0 in local coords
      # We want the base at the end of the shaft (at shaft_length)
      head_length = head_length_ratio * length
      head_position = start + direction * shaft_length
      head_positions[i] = head_position
      head_wxyzs[i] = rotation_quat
      head_scales[i] = [width, width, head_length]  # Per-axis scale
      head_colors[i] = (np.array(color[:3]) * 255).astype(
        np.uint8
      )  # Convert 0-1 to 0-255

    # Check if we need to recreate handles (number of arrows changed)
    needs_recreation = (
      self._arrow_shaft_handle is None
      or self._arrow_head_handle is None
      or len(shaft_positions) != len(self._arrow_shaft_handle.batched_positions)
    )

    if needs_recreation:
      # Remove old handles
      if self._arrow_shaft_handle is not None:
        self._arrow_shaft_handle.remove()
      if self._arrow_head_handle is not None:
        self._arrow_head_handle.remove()

      # Create new batched meshes
      self._arrow_shaft_handle = self.server.scene.add_batched_meshes_simple(
        f"/debug/env_{self.env_idx}/arrow_shafts",
        self._arrow_shaft_mesh.vertices,
        self._arrow_shaft_mesh.faces,
        batched_wxyzs=shaft_wxyzs,
        batched_positions=shaft_positions,
        batched_scales=shaft_scales,
        batched_colors=shaft_colors,
        opacity=0.5,
        cast_shadow=False,
        receive_shadow=False,
      )

      self._arrow_head_handle = self.server.scene.add_batched_meshes_simple(
        f"/debug/env_{self.env_idx}/arrow_heads",
        self._arrow_head_mesh.vertices,
        self._arrow_head_mesh.faces,
        batched_wxyzs=head_wxyzs,
        batched_positions=head_positions,
        batched_scales=head_scales,
        batched_colors=head_colors,
        opacity=0.5,
        cast_shadow=False,
        receive_shadow=False,
      )
    else:
      # Update existing handles (guaranteed to exist by needs_recreation check)
      assert self._arrow_shaft_handle is not None
      assert self._arrow_head_handle is not None

      self._arrow_shaft_handle.batched_positions = shaft_positions
      self._arrow_shaft_handle.batched_wxyzs = shaft_wxyzs
      self._arrow_shaft_handle.batched_scales = shaft_scales
      self._arrow_shaft_handle.batched_colors = shaft_colors

      self._arrow_head_handle.batched_positions = head_positions
      self._arrow_head_handle.batched_wxyzs = head_wxyzs
      self._arrow_head_handle.batched_scales = head_scales
      self._arrow_head_handle.batched_colors = head_colors

  @override
  def clear(self) -> None:
    """Clear all debug visualizations.

    Clears the arrow queue. Ghost meshes are kept and pose-updated for efficiency
    within the same environment, but removed when switching environments.
    """
    self._queued_arrows.clear()

  def clear_all(self) -> None:
    """Clear all debug visualizations including ghosts.

    Called when switching to a different environment.
    """
    self.clear()

    # Remove arrow meshes
    if self._arrow_shaft_handle is not None:
      self._arrow_shaft_handle.remove()
      self._arrow_shaft_handle = None
    if self._arrow_head_handle is not None:
      self._arrow_head_handle.remove()
      self._arrow_head_handle = None

    # Remove ghost meshes
    for handle in self._ghost_handles.values():
      handle.remove()
    self._ghost_handles.clear()

  @staticmethod
  def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (wxyz)."""
    return vtf.SO3.from_matrix(mat).wxyz
