"""Bridge between mjviser's ViserMujocoScene and mjlab's DebugVisualizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import trimesh
import viser
import viser.transforms as vtf
from mjviser import ViserMujocoScene
from mjviser.conversions import (
  create_primitive_mesh,
  mujoco_mesh_to_trimesh,
)
from mujoco import mjtGeom
from typing_extensions import override

from mjlab.viewer.debug_visualizer import DebugVisualizer

_Z_AXIS = np.array([0.0, 0.0, 1.0])
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _rotation_quat(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
  """Quaternion (wxyz) that rotates ``from_vec`` to ``to_vec``."""
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)
  if np.allclose(from_vec, to_vec):
    return _IDENTITY_QUAT.copy()
  if np.allclose(from_vec, -to_vec):
    perp = np.array([1.0, 0.0, 0.0])
    if abs(from_vec[0]) > 0.9:
      perp = np.array([0.0, 1.0, 0.0])
    axis = np.cross(from_vec, perp)
    axis = axis / np.linalg.norm(axis)
    return np.array([0.0, axis[0], axis[1], axis[2]])
  cross = np.cross(from_vec, to_vec)
  dot = np.dot(from_vec, to_vec)
  quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
  return quat / np.linalg.norm(quat)


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
  if isinstance(x, torch.Tensor):
    return x.cpu().numpy()
  return x


def _color_uint8(rgba: tuple[float, float, float, float]) -> np.ndarray:
  return (np.array(rgba[:3]) * 255).astype(np.uint8)


# Batched primitive handle.


@dataclass
class _BatchedPrimitive:
  """Manages a single batched mesh handle with lazy mesh creation."""

  name: str
  mesh_factory: Callable[[], trimesh.Trimesh]
  mesh: trimesh.Trimesh | None = field(default=None, repr=False)
  handle: viser.BatchedMeshHandle | None = field(default=None, repr=False)

  def remove(self) -> None:
    if self.handle is not None:
      self.handle.remove()
      self.handle = None

  def sync(
    self,
    server: viser.ViserServer,
    env_idx: int,
    positions: np.ndarray,
    wxyzs: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    opacity: float = 1.0,
  ) -> None:
    """Create or update the batched mesh handle."""
    if self.mesh is None:
      self.mesh = self.mesh_factory()

    needs_recreation = self.handle is None or len(positions) != len(
      self.handle.batched_positions
    )
    if needs_recreation:
      self.remove()
      self.handle = server.scene.add_batched_meshes_simple(
        f"/debug/env_{env_idx}/{self.name}",
        self.mesh.vertices,
        self.mesh.faces,
        batched_wxyzs=wxyzs,
        batched_positions=positions,
        batched_scales=scales,
        batched_colors=colors,
        opacity=opacity,
        cast_shadow=False,
        receive_shadow=False,
      )
    else:
      assert self.handle is not None
      self.handle.batched_positions = positions
      self.handle.batched_wxyzs = wxyzs
      self.handle.batched_scales = scales
      self.handle.batched_colors = colors


class MjlabViserScene(ViserMujocoScene, DebugVisualizer):
  """ViserMujocoScene with debug visualization and warp tensor conversion.

  Adds debug primitives (arrows, ghosts, spheres, cylinders, ellipsoids,
  coordinate frames) on top of the base scene from mjviser.
  """

  def __init__(
    self,
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    num_envs: int,
  ) -> None:
    super().__init__(server, mj_model, num_envs)

    self.debug_visualization_enabled = False
    self.show_all_envs = False

    # Queued debug primitives (populated each frame, consumed by sync).
    self._queued_arrows: list = []
    self._queued_ghosts: list = []
    self._queued_spheres: list = []
    self._queued_cylinders: list = []
    self._queued_ellipsoids: list = []

    # Batched mesh handles for simple primitives.
    def _shaft_mesh() -> trimesh.Trimesh:
      m = trimesh.creation.cylinder(radius=1.0, height=1.0)
      m.apply_translation(np.array([0, 0, 0.5]))
      return m

    self._arrow_shafts = _BatchedPrimitive("arrow_shafts", _shaft_mesh)
    self._arrow_heads = _BatchedPrimitive(
      "arrow_heads",
      lambda: trimesh.creation.cone(radius=2.0, height=1.0),
    )
    self._spheres = _BatchedPrimitive(
      "spheres",
      lambda: trimesh.creation.icosphere(subdivisions=2, radius=1.0),
    )
    self._cylinders = _BatchedPrimitive(
      "cylinders",
      lambda: trimesh.creation.cylinder(radius=1.0, height=1.0),
    )
    self._ellipsoids = _BatchedPrimitive(
      "ellipsoids",
      lambda: trimesh.creation.icosphere(subdivisions=2, radius=1.0),
    )
    self._all_primitives = [
      self._arrow_shafts,
      self._arrow_heads,
      self._spheres,
      self._cylinders,
      self._ellipsoids,
    ]

    # Ghost mesh state.
    self._ghost_handles: dict[tuple[int, int], viser.BatchedMeshHandle] = {}
    self._ghost_meshes: dict[int, dict[int, trimesh.Trimesh]] = {}

    # MjData used for ghost forward kinematics.
    self._viz_data = mujoco.MjData(mj_model)

  # Properties.

  @property
  @override
  def meansize(self) -> float:
    return self.meansize_override or self.mj_model.stat.meansize

  # Update entry points.

  def update(self, wp_data, env_idx: int | None = None) -> None:
    """Update scene from batched mjwarp simulation data.

    Converts warp GPU tensors to numpy arrays and delegates to
    ``update_from_arrays``.
    """
    body_xpos = wp_data.xpos.cpu().numpy()
    body_xmat = wp_data.xmat.cpu().numpy()
    if self.mj_model.nmocap > 0:
      mocap_pos = wp_data.mocap_pos.cpu().numpy()
      mocap_quat = wp_data.mocap_quat.cpu().numpy()
    else:
      mocap_pos = None
      mocap_quat = None

    kwargs: dict[str, np.ndarray] = {}
    if self._any_decor_visible():
      kwargs["qpos"] = wp_data.qpos.cpu().numpy()
      kwargs["qvel"] = wp_data.qvel.cpu().numpy()
      if self.mj_model.nu > 0:
        kwargs["ctrl"] = wp_data.ctrl.cpu().numpy()

    self.update_from_arrays(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      **kwargs,
    )

  @override
  def update_from_arrays(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray | None = None,
    mocap_quat: np.ndarray | None = None,
    env_idx: int | None = None,
    qpos: np.ndarray | None = None,
    qvel: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
  ) -> None:
    """Update scene and sync debug visualizations."""
    super().update_from_arrays(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      qpos=qpos,
      qvel=qvel,
      ctrl=ctrl,
    )
    self._sync_debug_visualizations(self._scene_offset)

  @override
  def update_from_mjdata(self, mj_data: mujoco.MjData) -> None:
    """Update scene and sync debug visualizations."""
    super().update_from_mjdata(mj_data)
    self._sync_debug_visualizations(self._scene_offset)

  # Refresh.

  @override
  def refresh_visualization(self) -> None:
    """Re-render, keeping needs_update set when debug viz is active."""
    super().refresh_visualization()
    self._sync_debug_visualizations(self._scene_offset)
    if self.debug_visualization_enabled:
      self.needs_update = True

  # GUI.

  @override
  def create_scene_gui(
    self,
    camera_distance: float = 3.0,
    camera_azimuth: float = 45.0,
    camera_elevation: float = 30.0,
    show_debug_viz_control: bool = True,
    debug_viz_extra_gui: Callable[[], None] | None = None,
  ) -> None:
    """Add standard GUI controls plus debug visualization section."""
    super().create_scene_gui(
      camera_distance=camera_distance,
      camera_azimuth=camera_azimuth,
      camera_elevation=camera_elevation,
    )

    if show_debug_viz_control:
      with self.server.gui.add_folder("Debug Viz"):
        cb_debug_vis = self.server.gui.add_checkbox(
          "Enabled",
          initial_value=self.debug_visualization_enabled,
          hint="Show debug arrows and ghost meshes.",
        )

        @cb_debug_vis.on_update
        def _(_) -> None:
          self.debug_visualization_enabled = cb_debug_vis.value
          if not self.debug_visualization_enabled:
            self.clear_debug_all()
          self.request_update()

        cb_show_all_envs = self.server.gui.add_checkbox(
          "All envs",
          initial_value=self.show_all_envs,
          hint="Show debug visualization for all environments.",
        )

        @cb_show_all_envs.on_update
        def _(_) -> None:
          self.show_all_envs = cb_show_all_envs.value
          if not self.show_all_envs:
            self.clear_debug_all()
          self.request_update()

        if debug_viz_extra_gui is not None:
          debug_viz_extra_gui()

  # DebugVisualizer ABC implementation.

  @override
  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    if not self.debug_visualization_enabled:
      return
    del label
    start, end = _to_numpy(start), _to_numpy(end)
    if np.linalg.norm(end - start) < 1e-6:
      return
    self._queued_arrows.append((start, end, color, width))

  @override
  def add_ghost_mesh(
    self,
    qpos: np.ndarray | torch.Tensor,
    model: mujoco.MjModel,
    mocap_pos: np.ndarray | torch.Tensor | None = None,
    mocap_quat: np.ndarray | torch.Tensor | None = None,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    if not self.debug_visualization_enabled:
      return
    qpos = _to_numpy(qpos)
    mocap_pos = _to_numpy(mocap_pos) if mocap_pos is not None else None
    mocap_quat = _to_numpy(mocap_quat) if mocap_quat is not None else None
    self._queued_ghosts.append(
      (
        qpos.copy(),
        model,
        np.asarray(mocap_pos).copy() if mocap_pos is not None else None,
        np.asarray(mocap_quat).copy() if mocap_quat is not None else None,
        alpha,
        label or f"env_{self.env_idx}",
      )
    )

  @override
  def add_frame(
    self,
    position: np.ndarray | torch.Tensor,
    rotation_matrix: np.ndarray | torch.Tensor,
    scale: float = 0.3,
    label: str | None = None,
    axis_radius: float = 0.01,
    alpha: float = 1.0,
    axis_colors: (tuple[tuple[float, float, float], ...] | None) = None,
  ) -> None:
    if not self.debug_visualization_enabled:
      return
    del label
    position = _to_numpy(position)
    rotation_matrix = _to_numpy(rotation_matrix)
    colors = axis_colors or [(0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]
    for axis_idx in range(3):
      end = position + rotation_matrix[:, axis_idx] * scale
      rgb = colors[axis_idx]
      self.add_arrow(
        start=position,
        end=end,
        color=(rgb[0], rgb[1], rgb[2], alpha),
        width=axis_radius,
      )

  @override
  def add_sphere(
    self,
    center: np.ndarray | torch.Tensor,
    radius: float,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    if not self.debug_visualization_enabled:
      return
    del label
    self._queued_spheres.append((_to_numpy(center).copy(), radius, color))

  @override
  def add_cylinder(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    radius: float,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    if not self.debug_visualization_enabled:
      return
    del label
    start, end = _to_numpy(start), _to_numpy(end)
    self._queued_cylinders.append((start.copy(), end.copy(), radius, color))

  @override
  def add_ellipsoid(
    self,
    center: np.ndarray | torch.Tensor,
    size: np.ndarray | torch.Tensor,
    mat: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    if not self.debug_visualization_enabled:
      return
    del label
    self._queued_ellipsoids.append(
      (
        np.asarray(_to_numpy(center), dtype=np.float32).copy(),
        np.asarray(_to_numpy(size), dtype=np.float32).copy(),
        np.asarray(_to_numpy(mat), dtype=np.float32).reshape(3, 3).copy(),
        color,
      )
    )

  @override
  def clear(self) -> None:
    """Clear all debug visualization queues."""
    self._queued_arrows.clear()
    self._queued_spheres.clear()
    self._queued_cylinders.clear()
    self._queued_ellipsoids.clear()
    self._queued_ghosts.clear()

  def clear_debug_all(self) -> None:
    """Clear all debug visualizations including handles."""
    self.clear()
    for prim in self._all_primitives:
      prim.remove()
    for handle in self._ghost_handles.values():
      handle.visible = False

  # Debug sync.

  def _sync_debug_visualizations(self, scene_offset: np.ndarray) -> None:
    if not self.debug_visualization_enabled:
      return
    self._scene_offset = scene_offset
    self._sync_arrows()
    self._sync_simple_primitives()
    self._sync_ghosts()

  def _sync_arrows(self) -> None:
    if not self._queued_arrows:
      self._arrow_shafts.remove()
      self._arrow_heads.remove()
      return

    n = len(self._queued_arrows)
    shaft_pos = np.zeros((n, 3), dtype=np.float32)
    shaft_wxyz = np.zeros((n, 4), dtype=np.float32)
    shaft_scale = np.zeros((n, 3), dtype=np.float32)
    shaft_col = np.zeros((n, 3), dtype=np.uint8)
    head_pos = np.zeros((n, 3), dtype=np.float32)
    head_wxyz = np.zeros((n, 4), dtype=np.float32)
    head_scale = np.zeros((n, 3), dtype=np.float32)
    head_col = np.zeros((n, 3), dtype=np.uint8)

    for i, (start, end, color, width) in enumerate(self._queued_arrows):
      s = start + self._scene_offset
      e = end + self._scene_offset
      d = e - s
      length = np.linalg.norm(d)
      d = d / length
      q = _rotation_quat(_Z_AXIS, d)
      c = _color_uint8(color)

      shaft_len = 0.8 * length
      shaft_pos[i] = s
      shaft_wxyz[i] = q
      shaft_scale[i] = [width, width, shaft_len]
      shaft_col[i] = c

      head_pos[i] = s + d * shaft_len
      head_wxyz[i] = q
      head_scale[i] = [width, width, 0.2 * length]
      head_col[i] = c

    self._arrow_shafts.sync(
      self.server,
      self.env_idx,
      shaft_pos,
      shaft_wxyz,
      shaft_scale,
      shaft_col,
    )
    self._arrow_heads.sync(
      self.server,
      self.env_idx,
      head_pos,
      head_wxyz,
      head_scale,
      head_col,
    )

  def _sync_simple_primitives(self) -> None:
    self._sync_spheres()
    self._sync_cylinders()
    self._sync_ellipsoids()

  def _sync_spheres(self) -> None:
    if not self._queued_spheres:
      self._spheres.remove()
      return
    n = len(self._queued_spheres)
    positions = np.zeros((n, 3), dtype=np.float32)
    wxyzs = np.tile(_IDENTITY_QUAT, (n, 1)).astype(np.float32)
    scales = np.zeros((n, 3), dtype=np.float32)
    colors = np.zeros((n, 3), dtype=np.uint8)
    opacity = 1.0
    for i, (center, radius, color) in enumerate(self._queued_spheres):
      positions[i] = center + self._scene_offset
      scales[i] = radius
      colors[i] = _color_uint8(color)
      opacity = color[3]
    self._spheres.sync(
      self.server,
      self.env_idx,
      positions,
      wxyzs,
      scales,
      colors,
      opacity,
    )

  def _sync_cylinders(self) -> None:
    if not self._queued_cylinders:
      self._cylinders.remove()
      return
    n = len(self._queued_cylinders)
    positions = np.zeros((n, 3), dtype=np.float32)
    wxyzs = np.zeros((n, 4), dtype=np.float32)
    scales = np.zeros((n, 3), dtype=np.float32)
    colors = np.zeros((n, 3), dtype=np.uint8)
    opacity = 1.0
    for i, (start, end, radius, color) in enumerate(self._queued_cylinders):
      s = start + self._scene_offset
      e = end + self._scene_offset
      d = e - s
      length = np.linalg.norm(d)
      if length < 1e-6:
        positions[i] = s
        wxyzs[i] = _IDENTITY_QUAT
      else:
        positions[i] = (s + e) / 2
        wxyzs[i] = _rotation_quat(_Z_AXIS, d / length)
        scales[i] = [radius, radius, length]
      colors[i] = _color_uint8(color)
      opacity = color[3]
    self._cylinders.sync(
      self.server,
      self.env_idx,
      positions,
      wxyzs,
      scales,
      colors,
      opacity,
    )

  def _sync_ellipsoids(self) -> None:
    if not self._queued_ellipsoids:
      self._ellipsoids.remove()
      return
    n = len(self._queued_ellipsoids)
    positions = np.zeros((n, 3), dtype=np.float32)
    wxyzs = np.zeros((n, 4), dtype=np.float32)
    scales = np.zeros((n, 3), dtype=np.float32)
    colors = np.zeros((n, 3), dtype=np.uint8)
    opacity = 1.0
    for i, (center, size, mat, color) in enumerate(self._queued_ellipsoids):
      positions[i] = center + self._scene_offset
      wxyzs[i] = vtf.SO3.from_matrix(mat).wxyz
      scales[i] = size
      colors[i] = _color_uint8(color)
      opacity = color[3]
    self._ellipsoids.sync(
      self.server,
      self.env_idx,
      positions,
      wxyzs,
      scales,
      colors,
      opacity,
    )

  def _sync_ghosts(self) -> None:
    """Render queued ghosts as one batched handle per (model, body)."""
    if not self._queued_ghosts:
      for handle in self._ghost_handles.values():
        handle.visible = False
      return

    body_data: dict[
      tuple[int, int],
      list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ] = {}
    alpha_by_model: dict[int, float] = {}

    for qpos, model, mocap_pos, mocap_quat, alpha, _label in self._queued_ghosts:
      model_id = id(model)
      alpha_by_model[model_id] = alpha

      # Forward kinematics on the visualization-only MjData.
      self._viz_data.qpos[:] = qpos
      if mocap_pos is not None and model.nmocap > 0:
        if mocap_pos.ndim == 1:
          self._viz_data.mocap_pos[0] = mocap_pos
        else:
          self._viz_data.mocap_pos[:] = mocap_pos
      if mocap_quat is not None and model.nmocap > 0:
        if mocap_quat.ndim == 1:
          self._viz_data.mocap_quat[0] = mocap_quat
        else:
          self._viz_data.mocap_quat[:] = mocap_quat
      mujoco.mj_forward(model, self._viz_data)

      # Group visible ghost geoms by body.
      body_geoms: dict[int, list[int]] = {}
      for gi in range(model.ngeom):
        if model.geom_rgba[gi, 3] == 0:
          continue
        bid = model.geom_bodyid[gi]
        if model.body_dofnum[bid] == 0 and model.body_parentid[bid] == 0:
          continue
        body_geoms.setdefault(bid, []).append(gi)

      for bid, bid_geom_ids in body_geoms.items():
        key = (model_id, bid)
        body_data.setdefault(key, []).append(
          (
            (self._viz_data.xpos[bid] + self._scene_offset).copy(),
            vtf.SO3.from_matrix(self._viz_data.xmat[bid].reshape(3, 3)).wxyz.copy(),
            (model.geom_rgba[bid_geom_ids[0]][:3] * 255).astype(np.uint8),
          )
        )

        # Cache combined mesh per (model, body).
        by_model = self._ghost_meshes.setdefault(model_id, {})
        if bid not in by_model:
          meshes = []
          for gid in bid_geom_ids:
            mesh = _create_geom_mesh(model, gid)
            if mesh is not None:
              T = np.eye(4)
              T[:3, :3] = vtf.SO3(model.geom_quat[gid]).as_matrix()
              T[:3, 3] = model.geom_pos[gid]
              mesh.apply_transform(T)
              meshes.append(mesh)
          if meshes:
            by_model[bid] = (
              meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
            )

    # Remove stale handles.
    for key in set(self._ghost_handles) - set(body_data):
      self._ghost_handles.pop(key).remove()

    # Create or update handles.
    for (model_id, bid), transforms in body_data.items():
      mesh = self._ghost_meshes.get(model_id, {}).get(bid)
      if mesh is None:
        continue

      positions = np.array([t[0] for t in transforms], dtype=np.float32)
      wxyzs = np.array([t[1] for t in transforms], dtype=np.float32)
      colors = np.array([t[2] for t in transforms], dtype=np.uint8)
      alpha = alpha_by_model.get(model_id, 0.5)
      key = (model_id, bid)

      if key not in self._ghost_handles:
        self._ghost_handles[key] = self.server.scene.add_batched_meshes_simple(
          f"/debug/ghosts/body_{bid}_{model_id}",
          mesh.vertices,
          mesh.faces,
          batched_wxyzs=wxyzs,
          batched_positions=positions,
          batched_colors=colors,
          opacity=alpha,
          cast_shadow=False,
          receive_shadow=False,
        )
      else:
        handle = self._ghost_handles[key]
        try:
          handle.batched_positions = positions
          handle.batched_wxyzs = wxyzs
          handle.batched_colors = colors
          handle.visible = True
        except Exception:
          handle.remove()
          self._ghost_handles[key] = self.server.scene.add_batched_meshes_simple(
            f"/debug/ghosts/body_{bid}_{model_id}",
            mesh.vertices,
            mesh.faces,
            batched_wxyzs=wxyzs,
            batched_positions=positions,
            batched_colors=colors,
            opacity=alpha,
            cast_shadow=False,
            receive_shadow=False,
          )


# Helpers.


def _create_geom_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
  if mj_model.geom_type[geom_id] == mjtGeom.mjGEOM_MESH:
    return mujoco_mesh_to_trimesh(mj_model, geom_id)
  return create_primitive_mesh(mj_model, geom_id)
