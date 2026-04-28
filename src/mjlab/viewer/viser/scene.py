"""Bridge between mjviser's ViserMujocoScene and mjlab's DebugVisualizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import mujoco
import numpy as np
import torch
import trimesh
import viser
import viser.transforms as vtf
from mjviser import ViserMujocoScene
from mjviser.conversions import (
  create_primitive_mesh,
  get_body_name,
  group_geoms_by_visual_compat,
  is_fixed_body,
  merge_geoms,
  mujoco_mesh_to_trimesh,
)
from mujoco import mjtGeom
from typing_extensions import override

from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.model_sync import (
  VIEWER_MODEL_FIELDS,
  disable_model_sameframe_shortcuts,
  sync_model_fields,
)

_Z_AXIS = np.array([0.0, 0.0, 1.0])
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_VISER_GEOMETRY_HANDLE_FIELDS = frozenset(
  {
    "geom_dataid",
    "geom_size",
    "geom_pos",
    "geom_quat",
  }
)
_VISER_APPEARANCE_HANDLE_FIELDS = frozenset(
  {
    "geom_rgba",
    "mat_rgba",
  }
)
_VISER_BAKED_HANDLE_FIELDS = (
  _VISER_GEOMETRY_HANDLE_FIELDS | _VISER_APPEARANCE_HANDLE_FIELDS
)


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


@dataclass
class _PerWorldMeshGroup:
  handle: viser.BatchedGlbHandle
  body_ids: np.ndarray
  group_id: int
  mocap_ids: np.ndarray | None
  env_ids: np.ndarray


@dataclass
class _VariantMeshGroup:
  mesh: trimesh.Trimesh
  group_id: int
  sub_idx: int
  body_name: str
  is_mocap: bool
  env_ids: list[int] = field(default_factory=list)
  body_ids: list[int] = field(default_factory=list)
  mocap_ids: list[int] = field(default_factory=list)


@dataclass
class _PerWorldHullGroup:
  """One convex-hull handle covering the envs that share a hull variant."""

  handle: viser.BatchedMeshHandle
  body_id: int
  env_ids: np.ndarray


@dataclass
class _HullVariant:
  body_id: int
  vertices: np.ndarray
  faces: np.ndarray
  env_ids: list[int] = field(default_factory=list)


def _compute_body_hull(
  mj_model: mujoco.MjModel, geom_ids: list[int]
) -> trimesh.Trimesh | None:
  """Compute a merged convex hull for a body's mesh geoms.

  Bypasses ``mjviser.merge_geoms_hull`` (which reads ``mesh_polynum`` and
  returns ``None`` for any mesh that MuJoCo didn't compile polygon data for --
  a situation that arises for later per-world-mesh variants). Uses the raw
  ``mesh_vert`` / ``mesh_face`` arrays and trimesh's convex hull instead.
  """
  pieces: list[trimesh.Trimesh] = []
  for geom_id in geom_ids:
    if int(mj_model.geom_type[geom_id]) != int(mjtGeom.mjGEOM_MESH):
      continue
    mesh_id = int(mj_model.geom_dataid[geom_id])
    if mesh_id < 0:
      continue
    vert_start = int(mj_model.mesh_vertadr[mesh_id])
    vert_count = int(mj_model.mesh_vertnum[mesh_id])
    if vert_count < 4:
      continue
    vertices = mj_model.mesh_vert[vert_start : vert_start + vert_count].copy()
    try:
      piece = trimesh.PointCloud(vertices).convex_hull
    except Exception:
      continue
    transform = np.eye(4)
    transform[:3, :3] = vtf.SO3(mj_model.geom_quat[geom_id]).as_matrix()
    transform[:3, 3] = mj_model.geom_pos[geom_id]
    piece.apply_transform(transform)
    pieces.append(piece)
  if not pieces:
    return None
  merged = pieces[0] if len(pieces) == 1 else trimesh.util.concatenate(pieces)
  try:
    return merged.convex_hull
  except Exception:
    return merged


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
    sim_model: Any | None = None,
    expanded_fields: set[str] | None = None,
  ) -> None:
    self._sim_model = sim_model
    self._expanded_fields = expanded_fields or set()
    self._baked_appearance_fields = (
      self._expanded_fields & _VISER_APPEARANCE_HANDLE_FIELDS
    )
    self._baked_appearance_fingerprint: tuple[tuple[str, bytes], ...] | None = None
    self._use_per_world_mesh_groups = bool(
      self._expanded_fields & _VISER_BAKED_HANDLE_FIELDS
    )
    # Populated by _build_hull_handles when per-world variants are active.
    # Initialized here because ViserMujocoScene.__init__ calls our overrides
    # of _compute_hull_body_meshes / _build_hull_handles during super().__init__.
    self._hull_per_world_groups: list[_PerWorldHullGroup] = []
    if self._sim_model is not None:
      sync_model_fields(
        mj_model,
        self._sim_model,
        self._expanded_fields & VIEWER_MODEL_FIELDS,
        0,
      )
      disable_model_sameframe_shortcuts(mj_model)
    super().__init__(server, mj_model, num_envs)
    self._baked_appearance_fingerprint = self._appearance_fingerprint()

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
    if env_idx is None:
      env_idx = self.env_idx
    self._sync_model_fields(env_idx)
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
    self._sync_model_fields(self.env_idx)
    super().update_from_mjdata(mj_data)
    self._sync_debug_visualizations(self._scene_offset)

  def _sync_model_fields(self, env_idx: int) -> None:
    """Sync visually relevant per-world model fields into the host MjModel."""
    if self._sim_model is None:
      return
    fields = self._expanded_fields & VIEWER_MODEL_FIELDS
    sync_model_fields(self.mj_model, self._sim_model, fields, env_idx)
    self._rebuild_visual_handles_if_needed()

  def _appearance_fingerprint(self) -> tuple[tuple[str, bytes], ...] | None:
    """Return a stable fingerprint for fields baked into Viser mesh handles."""
    if self._sim_model is None or not self._baked_appearance_fields:
      return None
    parts: list[tuple[str, bytes]] = []
    for field_name in sorted(self._baked_appearance_fields):
      value = getattr(self._sim_model, field_name).cpu().numpy()
      parts.append((field_name, value.tobytes()))
    return tuple(parts)

  def _rebuild_visual_handles_if_needed(self) -> None:
    if self._baked_appearance_fingerprint is None:
      return
    fingerprint = self._appearance_fingerprint()
    if fingerprint == self._baked_appearance_fingerprint:
      return
    self._baked_appearance_fingerprint = fingerprint
    self.rebuild_visual_handles()

  @staticmethod
  def _geom_subgroup_visual_fingerprint(
    mj_model: mujoco.MjModel, geom_ids: list[int], is_mocap: bool
  ) -> tuple[object, ...]:
    parts: list[tuple[object, ...]] = []
    for geom_id in geom_ids:
      mat_id = int(mj_model.geom_matid[geom_id])
      mat_rgba = (
        tuple(mj_model.mat_rgba[mat_id].round(4).tolist()) if mat_id >= 0 else None
      )
      parts.append(
        (
          int(mj_model.geom_type[geom_id]),
          int(mj_model.geom_dataid[geom_id]),
          mat_id,
          mat_rgba,
          tuple(mj_model.geom_size[geom_id].round(6).tolist()),
          tuple(mj_model.geom_rgba[geom_id].round(4).tolist()),
          tuple(mj_model.geom_pos[geom_id].round(6).tolist()),
          tuple(mj_model.geom_quat[geom_id].round(6).tolist()),
        )
      )
    return (is_mocap, tuple(sorted(parts)))

  @override
  def _create_mesh_handles_by_group(self) -> None:
    """Create dynamic mesh handles, respecting per-world mesh variants."""
    if not self._use_per_world_mesh_groups:
      super()._create_mesh_handles_by_group()
      return

    variants: dict[tuple[object, ...], _VariantMeshGroup] = {}
    for env_idx in range(self.num_envs):
      self._sync_model_fields(env_idx)
      body_group_geoms: dict[tuple[int, int], list[int]] = {}
      for geom_id in range(self.mj_model.ngeom):
        body_id = int(self.mj_model.geom_bodyid[geom_id])
        if is_fixed_body(self.mj_model, body_id):
          continue
        if self.mj_model.geom_rgba[geom_id, 3] == 0:
          continue
        if (
          int(self.mj_model.geom_type[geom_id]) == int(mjtGeom.mjGEOM_MESH)
          and int(self.mj_model.geom_dataid[geom_id]) < 0
        ):
          continue
        group_id = int(self.mj_model.geom_group[geom_id])
        body_group_geoms.setdefault((body_id, group_id), []).append(geom_id)

      for (body_id, group_id), geom_ids in body_group_geoms.items():
        subgroups = group_geoms_by_visual_compat(self.mj_model, geom_ids)
        is_mocap = bool(self.mj_model.body_mocapid[body_id] >= 0)
        for sub_idx, sub_geom_ids in enumerate(subgroups):
          fp = self._geom_subgroup_visual_fingerprint(
            self.mj_model, sub_geom_ids, is_mocap
          )
          key = (fp, group_id, sub_idx)
          variant = variants.get(key)
          if variant is None:
            variant = _VariantMeshGroup(
              mesh=merge_geoms(self.mj_model, sub_geom_ids),
              group_id=group_id,
              sub_idx=sub_idx,
              body_name=get_body_name(self.mj_model, body_id),
              is_mocap=is_mocap,
            )
            variants[key] = variant
          variant.env_ids.append(env_idx)
          variant.body_ids.append(body_id)
          if is_mocap:
            variant.mocap_ids.append(int(self.mj_model.body_mocapid[body_id]))

    self._sync_model_fields(self.env_idx)
    with self.server.atomic():
      for variant_idx, variant in enumerate(variants.values()):
        batch_count = len(variant.env_ids)
        lod_ratio = 1000.0 / variant.mesh.vertices.shape[0]
        suffix = f"/sub{variant.sub_idx}" if variant.sub_idx > 0 else ""
        visible = variant.group_id < 6 and self.geom_groups_visible[variant.group_id]

        handle = self.server.scene.add_batched_meshes_trimesh(
          f"/bodies/{variant.body_name}/group{variant.group_id}"
          f"/variant{variant_idx}{suffix}",
          variant.mesh,
          batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (batch_count, 1)),
          batched_positions=np.zeros((batch_count, 3)),
          lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
          visible=visible,
        )
        cast(Any, self._mesh_groups).append(
          _PerWorldMeshGroup(
            handle=handle,
            body_ids=np.asarray(variant.body_ids, dtype=np.int32),
            group_id=variant.group_id,
            mocap_ids=(
              np.asarray(variant.mocap_ids, dtype=np.int32)
              if variant.is_mocap
              else None
            ),
            env_ids=np.asarray(variant.env_ids, dtype=np.int32),
          )
        )

  @override
  def _compute_hull_body_meshes(self) -> None:
    """Record hull-bearing bodies across all variants; meshes built lazily."""
    if not self._use_per_world_mesh_groups:
      super()._compute_hull_body_meshes()
      return
    # Upstream caches one merged hull per body from the current mj_model.
    # With per-world variants each env can have a different set of active
    # mesh slots, so the actual hulls are computed per-variant in
    # _build_hull_handles. Here we just record which bodies carry mesh hulls
    # in any env so mjviser's _sync_visibilities / _hull_hide_meshes logic
    # still has the right body set.
    self._hull_body_meshes = {}
    # Read the per-world geom_dataid table directly from sim_model: a body
    # is a hull body iff any world has at least one active mesh geom on it,
    # which is constant data we don't need to materialize per-env into
    # mj_model to inspect.
    assert self._sim_model is not None
    dataid = self._sim_model.geom_dataid.cpu().numpy()
    if dataid.ndim == 1:
      dataid = dataid[None, :]
    geom_active_in_any_world = (dataid >= 0).any(axis=0)
    hull_bodies: set[int] = set()
    for geom_id in range(self.mj_model.ngeom):
      if int(self.mj_model.geom_type[geom_id]) != int(mjtGeom.mjGEOM_MESH):
        continue
      if not geom_active_in_any_world[geom_id]:
        continue
      hull_bodies.add(int(self.mj_model.geom_bodyid[geom_id]))
    self._hull_mesh_bodies = hull_bodies

  @override
  def _build_hull_handles(self) -> None:
    """Build one batched hull handle per (body, variant) across envs."""
    if not self._use_per_world_mesh_groups:
      super()._build_hull_handles()
      return

    color = np.array(self._hull_color, dtype=np.uint8)
    opacity = float(self._hull_opacity)

    # Group envs by (body_id, hull fingerprint). Fingerprint captures the
    # fields that merge_geoms_hull actually reads (geom_dataid + local
    # geom_pos/quat), so any two envs with identical fingerprints share the
    # same hull mesh in body-local space.
    variants: dict[tuple[int, tuple[object, ...]], _HullVariant] = {}
    fixed_hull_bodies: dict[int, list[int]] = {}

    for env_idx in range(self.num_envs):
      self._sync_model_fields(env_idx)
      body_geoms: dict[int, list[int]] = {}
      for geom_id in range(self.mj_model.ngeom):
        if int(self.mj_model.geom_type[geom_id]) != int(mjtGeom.mjGEOM_MESH):
          continue
        if int(self.mj_model.geom_dataid[geom_id]) < 0:
          continue
        body_id = int(self.mj_model.geom_bodyid[geom_id])
        body_geoms.setdefault(body_id, []).append(geom_id)

      for body_id, geom_ids in body_geoms.items():
        if is_fixed_body(self.mj_model, body_id):
          # Fixed bodies don't need per-env batching; defer to upstream
          # single-hull path using env_idx 0 (already the default).
          if env_idx == 0:
            fixed_hull_bodies[body_id] = geom_ids
          continue
        fingerprint = tuple(
          (
            int(self.mj_model.geom_dataid[gid]),
            tuple(float(x) for x in self.mj_model.geom_pos[gid].round(6)),
            tuple(float(x) for x in self.mj_model.geom_quat[gid].round(6)),
          )
          for gid in geom_ids
        )
        key = (body_id, fingerprint)
        v = variants.get(key)
        if v is None:
          hull = _compute_body_hull(self.mj_model, geom_ids)
          if hull is None:
            continue
          v = _HullVariant(
            body_id=body_id,
            vertices=hull.vertices.astype(np.float32),
            faces=hull.faces.astype(np.int32),
          )
          variants[key] = v
        v.env_ids.append(env_idx)

    # Fixed bodies: build one hull handle each (same body-local mesh for all
    # envs since the body is welded). Uses env 0's synced state which is the
    # default after the loop below.
    self._sync_model_fields(self.env_idx)
    for body_id, geom_ids in fixed_hull_bodies.items():
      hull = _compute_body_hull(self.mj_model, geom_ids)
      if hull is None:
        continue
      body = self.mj_model.body(body_id)
      fixed_opacities = (
        None if opacity >= 1.0 else np.array([opacity], dtype=np.float32)
      )
      handle = self.server.scene.add_batched_meshes_simple(
        f"/fixed_bodies/hull/{body_id}",
        hull.vertices.astype(np.float32),
        hull.faces.astype(np.int32),
        batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        batched_positions=np.zeros((1, 3), dtype=np.float32),
        batched_colors=color[None],
        batched_opacities=fixed_opacities,
        position=body.pos,
        wxyz=body.quat,
        visible=self._show_convex_hull,
        cast_shadow=False,
        receive_shadow=False,
        lod="off",
      )
      self._hull_fixed_handles[body_id] = handle

    self._hull_dynamic_handles = []
    self._hull_per_world_groups = []
    for variant_idx, ((body_id, _fp), v) in enumerate(variants.items()):
      env_ids = np.asarray(v.env_ids, dtype=np.int32)
      batch_count = int(env_ids.size)
      dynamic_opacities = (
        None if opacity >= 1.0 else np.full(batch_count, opacity, dtype=np.float32)
      )
      handle = self.server.scene.add_batched_meshes_simple(
        f"/hull/{body_id}/variant{variant_idx}",
        v.vertices,
        v.faces,
        batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (batch_count, 1)).astype(
          np.float32
        ),
        batched_positions=np.zeros((batch_count, 3), dtype=np.float32),
        batched_colors=np.tile(color, (batch_count, 1)),
        batched_opacities=dynamic_opacities,
        visible=self._show_convex_hull,
        cast_shadow=False,
        receive_shadow=False,
        lod="off",
      )
      self._hull_per_world_groups.append(
        _PerWorldHullGroup(handle=handle, body_id=body_id, env_ids=env_ids)
      )
      # Also register in the upstream list so show_convex_hull.setter and
      # any other base-class consumer still see every dynamic hull handle.
      self._hull_dynamic_handles.append((handle, body_id))

  @override
  def _clear_hull_handles(self) -> None:
    super()._clear_hull_handles()
    self._hull_per_world_groups = []

  @override
  def _update_visualization_locked(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    mj_data: mujoco.MjData | None = None,
  ) -> None:
    if not self._use_per_world_mesh_groups:
      super()._update_visualization_locked(
        body_xpos, body_xmat, mocap_pos, mocap_quat, env_idx, scene_offset, mj_data
      )
      return

    self._last_body_xpos = body_xpos
    self._last_body_xmat = body_xmat
    self._last_mocap_pos = mocap_pos
    self._last_mocap_quat = mocap_quat
    self._last_env_idx = env_idx
    self._scene_offset = scene_offset
    if mj_data is not None:
      self._last_mj_data = mj_data

    self.fixed_bodies_frame.position = scene_offset
    slice_single = self.show_only_selected and self.num_envs > 1
    hidden_bodies: set[int] = set()
    if self._show_convex_hull and self._hull_hide_meshes:
      hidden_bodies = self._hull_mesh_bodies
    if (
      self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT]
      and self._autoconnect_hide_meshes
    ):
      hidden_bodies |= set(range(self.mj_model.nbody))

    with self.server.atomic():
      body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz
      for mg in self._mesh_groups:
        if isinstance(mg, _PerWorldMeshGroup):
          visible = mg.group_id < 6 and self.geom_groups_visible[mg.group_id]
          if visible and any(body_id in hidden_bodies for body_id in mg.body_ids):
            visible = False
          if not visible:
            mg.handle.visible = False
            continue

          env_ids = mg.env_ids
          body_ids = mg.body_ids
          mocap_ids = mg.mocap_ids
          if slice_single:
            mask = env_ids == env_idx
            if not np.any(mask):
              mg.handle.visible = False
              continue
            env_ids = env_ids[mask]
            body_ids = body_ids[mask]
            if mocap_ids is not None:
              mocap_ids = mocap_ids[mask]
          if mocap_ids is not None:
            pos = mocap_pos[env_ids, mocap_ids] + scene_offset
            quat = mocap_quat[env_ids, mocap_ids]
          else:
            pos = body_xpos[env_ids, body_ids] + scene_offset
            quat = body_xquat[env_ids, body_ids]
          mg.handle.batched_positions = pos
          mg.handle.batched_wxyzs = quat
          mg.handle.visible = True
          continue

        if not mg.handle.visible:
          continue
        if mg.mocap_ids is not None:
          pos, quat = self._batched_transform_group(
            mocap_pos, mocap_quat, mg.mocap_ids, env_idx, scene_offset, slice_single
          )
        else:
          pos, quat = self._batched_transform_group(
            body_xpos, body_xquat, mg.body_ids, env_idx, scene_offset, slice_single
          )
        mg.handle.batched_positions = pos
        mg.handle.batched_wxyzs = quat

      for (body_id, _), handle in self.site_handles_by_group.items():
        if not handle.visible:
          continue
        pos, quat = self._batched_transform(
          body_xpos, body_xquat, body_id, env_idx, scene_offset, slice_single
        )
        handle.batched_positions = pos
        handle.batched_wxyzs = quat

      if self._show_convex_hull:
        for hg in self._hull_per_world_groups:
          env_ids = hg.env_ids
          body_id = hg.body_id
          if slice_single:
            mask = env_ids == env_idx
            if not np.any(mask):
              hg.handle.visible = False
              continue
            env_ids = env_ids[mask]
          pos = body_xpos[env_ids, body_id] + scene_offset
          quat = body_xquat[env_ids, body_id]
          hg.handle.batched_positions = pos
          hg.handle.batched_wxyzs = quat
          hg.handle.visible = True

      if self._any_decor_visible() and mj_data is not None:
        self._update_decor_from_mjvscene(mj_data, scene_offset)
      elif not self._any_decor_visible():
        self._clear_decor_handles()

      self.server.flush()

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
