"""Terrains imported from external triangle mesh files."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import trimesh

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import find_flat_patches_from_heightfield

MeshCollisionMode = Literal["heightfield", "parts", "convex"]

_MIN_HFIELD_HEIGHT = 1e-3
_COLLISION_HFIELD_RGBA = (0.2, 0.6, 0.2, 0.3)
_DEFAULT_MESH_RGBA = (0.55, 0.55, 0.58, 1.0)


def load_terrain_meshes(
  mesh_path: str | Path,
  *,
  scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
  target_xy_center: tuple[float, float] | None = None,
  align_min_z: bool = False,
) -> tuple[list[trimesh.Trimesh], trimesh.Trimesh]:
  """Load a mesh or multi-part scene file and apply terrain transforms.

  Supports formats handled by ``trimesh`` (``.obj``, ``.stl``, ``.ply``, ``.glb``,
  ``.gltf``, etc.).

  Args:
    mesh_path: Path to the mesh file on disk.
    scale: Per-axis ``(sx, sy, sz)`` scale factors; all must be positive.
    target_xy_center: If provided, translates mesh vertices in XY so the combined
      bounding-box center lands at ``(cx, cy)``.
    align_min_z: If True, shifts vertices along Z so the minimum vertex Z is 0.0.

  Returns:
    Tuple ``(parts, combined)`` where ``parts`` is the list of individual
    ``trimesh.Trimesh`` objects in the file and ``combined`` is their concatenation.
  """
  path = Path(mesh_path)
  if not path.is_file():
    raise FileNotFoundError(f"Mesh terrain file not found: {path}")

  if len(scale) != 3 or any(s <= 0.0 for s in scale):
    raise ValueError(f"Mesh terrain scale must have 3 positive values, got {scale}.")

  loaded = trimesh.load(path, process=False)
  raw_parts: list[trimesh.Trimesh] = []
  if isinstance(loaded, trimesh.Scene):
    for geom in loaded.dump():
      if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
        raw_parts.append(geom.copy())
  elif isinstance(loaded, trimesh.Trimesh):
    if len(loaded.faces) > 0:
      raw_parts.append(loaded.copy())

  if not raw_parts:
    raise ValueError(f"Mesh terrain file '{path}' contains no triangle faces.")

  scale_arr = np.asarray(scale, dtype=np.float64)
  for part in raw_parts:
    # Merge coincident vertices (e.g. unindexed STL triangle soup) so connected
    # components remain intact when splitting in "parts" mode.
    part.merge_vertices()
    part.vertices = np.asarray(part.vertices, dtype=np.float64) * scale_arr

  combined = (
    raw_parts[0].copy() if len(raw_parts) == 1 else trimesh.util.concatenate(raw_parts)
  )
  bounds = np.asarray(combined.bounds, dtype=np.float64)

  offset = np.zeros(3, dtype=np.float64)
  if target_xy_center is not None:
    bbox_center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    offset[0] = target_xy_center[0] - bbox_center_xy[0]
    offset[1] = target_xy_center[1] - bbox_center_xy[1]
  if align_min_z:
    offset[2] = -bounds[0, 2]

  if np.any(offset != 0.0):
    for part in raw_parts:
      part.vertices = part.vertices + offset
    combined.vertices = combined.vertices + offset

  return raw_parts, combined


def query_mesh_top_z(
  vertices: np.ndarray,
  faces: np.ndarray,
  xy: np.ndarray,
  fallback_z: float = 0.0,
) -> np.ndarray:
  """Query the highest triangle surface Z at each ``(x, y)`` coordinate.

  Uses exact barycentric evaluation in NumPy with no external raytracing
  dependencies (such as ``rtree`` or ``pyembree``).

  Args:
    vertices: Array of shape ``(V, 3)``.
    faces: Integer triangle index array of shape ``(F, 3)``.
    xy: Query coordinates of shape ``(N, 2)``.
    fallback_z: Height returned for query points not covered by any non-vertical
      triangle.

  Returns:
    Array of shape ``(N,)`` with surface heights in meters.
  """
  xy_arr = np.atleast_2d(np.asarray(xy, dtype=np.float64))
  out = np.full(xy_arr.shape[0], fallback_z, dtype=np.float64)
  if faces.size == 0:
    return out

  tris = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
  v0 = tris[:, 0]
  v1 = tris[:, 1]
  v2 = tris[:, 2]

  dx1 = v1[:, 0] - v0[:, 0]
  dy1 = v1[:, 1] - v0[:, 1]
  dx2 = v2[:, 0] - v0[:, 0]
  dy2 = v2[:, 1] - v0[:, 1]
  det = dx1 * dy2 - dx2 * dy1

  valid_tri = np.abs(det) > 1e-12
  if not np.any(valid_tri):
    return out

  v0 = v0[valid_tri]
  v1 = v1[valid_tri]
  v2 = v2[valid_tri]
  dx1 = dx1[valid_tri]
  dy1 = dy1[valid_tri]
  dx2 = dx2[valid_tri]
  dy2 = dy2[valid_tri]
  inv_det = 1.0 / det[valid_tri]

  min_x = np.minimum(np.minimum(v0[:, 0], v1[:, 0]), v2[:, 0]) - 1e-9
  max_x = np.maximum(np.maximum(v0[:, 0], v1[:, 0]), v2[:, 0]) + 1e-9
  min_y = np.minimum(np.minimum(v0[:, 1], v1[:, 1]), v2[:, 1]) - 1e-9
  max_y = np.maximum(np.maximum(v0[:, 1], v1[:, 1]), v2[:, 1]) + 1e-9

  for idx in range(xy_arr.shape[0]):
    px, py = xy_arr[idx, 0], xy_arr[idx, 1]
    cand = np.flatnonzero((px >= min_x) & (px <= max_x) & (py >= min_y) & (py <= max_y))
    if cand.size == 0:
      continue
    rx = px - v0[cand, 0]
    ry = py - v0[cand, 1]
    u = (rx * dy2[cand] - ry * dx2[cand]) * inv_det[cand]
    v = (dx1[cand] * ry - dy1[cand] * rx) * inv_det[cand]
    w = 1.0 - u - v
    inside = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
    if np.any(inside):
      hit_c = cand[inside]
      z_vals = (
        w[inside] * v0[hit_c, 2] + u[inside] * v1[hit_c, 2] + v[inside] * v2[hit_c, 2]
      )
      out[idx] = float(np.max(z_vals))

  return out


def rasterize_mesh_heightfield(
  vertices: np.ndarray,
  faces: np.ndarray,
  x_bounds: tuple[float, float],
  y_bounds: tuple[float, float],
  horizontal_scale: float,
  default_z: float | None = None,
) -> tuple[np.ndarray, float, float]:
  """Rasterize the top surface of a triangle mesh onto a MuJoCo heightfield grid.

  Follows MuJoCo's heightfield axis convention: rows run along ``y`` and columns
  run along ``x``.

  Args:
    vertices: Mesh vertices of shape ``(V, 3)``.
    faces: Mesh triangle indices of shape ``(F, 3)``.
    x_bounds: ``(x_min, x_max)`` span of the heightfield in meters.
    y_bounds: ``(y_min, y_max)`` span of the heightfield in meters.
    horizontal_scale: Target grid cell spacing in meters (must be positive).
    default_z: Fill height for grid cells not covered by any triangle. Defaults to
      the mesh minimum Z.

  Returns:
    Tuple ``(heights, dx, dy)`` where ``heights`` has shape ``(nrow, ncol)`` and
    ``dx``, ``dy`` are the exact grid spacings along x and y.
  """
  if horizontal_scale <= 0.0:
    raise ValueError(f"horizontal_scale must be positive, got {horizontal_scale}.")

  x_min, x_max = x_bounds
  y_min, y_max = y_bounds
  span_x = max(x_max - x_min, _MIN_HFIELD_HEIGHT)
  span_y = max(y_max - y_min, _MIN_HFIELD_HEIGHT)

  ncol = max(2, int(np.ceil(span_x / horizontal_scale)) + 1)
  nrow = max(2, int(np.ceil(span_y / horizontal_scale)) + 1)
  dx = span_x / (ncol - 1)
  dy = span_y / (nrow - 1)

  verts = np.asarray(vertices, dtype=np.float64)
  tris = verts[np.asarray(faces, dtype=np.int64)]
  heights = np.full((nrow, ncol), -np.inf, dtype=np.float64)

  xs = np.linspace(x_min, x_min + span_x, ncol)
  ys = np.linspace(y_min, y_min + span_y, nrow)

  for tri in tris:
    v0, v1, v2 = tri[0], tri[1], tri[2]
    dx1 = v1[0] - v0[0]
    dy1 = v1[1] - v0[1]
    dx2 = v2[0] - v0[0]
    dy2 = v2[1] - v0[1]
    det = dx1 * dy2 - dx2 * dy1
    if abs(det) <= 1e-12:
      continue

    t_xmin = min(v0[0], v1[0], v2[0])
    t_xmax = max(v0[0], v1[0], v2[0])
    t_ymin = min(v0[1], v1[1], v2[1])
    t_ymax = max(v0[1], v1[1], v2[1])

    c0 = max(0, int(np.floor((t_xmin - x_min) / dx)))
    c1 = min(ncol - 1, int(np.ceil((t_xmax - x_min) / dx)))
    r0 = max(0, int(np.floor((t_ymin - y_min) / dy)))
    r1 = min(nrow - 1, int(np.ceil((t_ymax - y_min) / dy)))
    if c0 > c1 or r0 > r1:
      continue

    rx = xs[c0 : c1 + 1][None, :] - v0[0]
    ry = ys[r0 : r1 + 1][:, None] - v0[1]
    inv_det = 1.0 / det
    u = (rx * dy2 - ry * dx2) * inv_det
    v = (dx1 * ry - dy1 * rx) * inv_det
    w = 1.0 - u - v
    inside = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
    if not np.any(inside):
      continue

    z_patch = w * v0[2] + u * v1[2] + v * v2[2]
    sub = heights[r0 : r1 + 1, c0 : c1 + 1]
    sub[inside] = np.maximum(sub[inside], z_patch[inside])

  fill_z = float(np.min(verts[:, 2])) if default_z is None else float(default_z)
  heights = np.where(np.isfinite(heights), heights, fill_z)
  return heights, dx, dy


def _ensure_volumetric_mesh(
  vertices: np.ndarray, faces: np.ndarray, thickness: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
  """Ensure a collision mesh has >=4 non-coplanar vertices for MuJoCo Qhull."""
  centered = vertices - vertices[0]
  if vertices.shape[0] >= 4 and np.linalg.matrix_rank(centered, tol=1e-6) == 3:
    return vertices, faces
  # Extrude coplanar/degenerate sheet slightly downward so Qhull builds a valid
  # 3D hull while keeping the top collision surface unchanged.
  extruded_v = np.vstack([vertices, vertices - np.array([0.0, 0.0, thickness])])
  extruded_f = np.vstack([faces, faces[:, ::-1] + vertices.shape[0]])
  return extruded_v, extruded_f


def build_mesh_terrain_geometries(
  spec: mujoco.MjSpec,
  body: mujoco.MjsBody,
  parts: list[trimesh.Trimesh],
  combined: trimesh.Trimesh,
  *,
  collision_mode: MeshCollisionMode,
  horizontal_scale: float,
  xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
  geom_name: str | None = None,
  rgba: tuple[float, float, float, float] = _DEFAULT_MESH_RGBA,
) -> tuple[list[TerrainGeometry], np.ndarray | None, float]:
  """Attach visual and collision geometries for an imported mesh to ``body``.

  Returns:
    Tuple ``(geometries, raster_heights, raster_dx)`` where ``raster_heights`` is
    populated when ``collision_mode == "heightfield"`` (else ``None``).
  """
  uid = uuid.uuid4().hex
  geometries: list[TerrainGeometry] = []
  verts = np.asarray(combined.vertices, dtype=np.float64)
  faces = np.asarray(combined.faces, dtype=np.int32)

  if collision_mode == "heightfield":
    mesh_asset = spec.add_mesh(
      name=f"mesh_terrain_{uid}",
      uservert=verts.reshape(-1).tolist(),
      userface=faces.reshape(-1).tolist(),
    )
    vis_geom = body.add_geom(
      name=geom_name,
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_asset.name,
      group=0,
      contype=0,
      conaffinity=0,
      rgba=rgba,
    )
    geometries.append(TerrainGeometry(geom=vis_geom, color=rgba))

    bounds = np.asarray(combined.bounds, dtype=np.float64)
    if xy_bounds is not None:
      x_bounds, y_bounds = xy_bounds
    else:
      x_bounds = (float(bounds[0, 0]), float(bounds[1, 0]))
      y_bounds = (float(bounds[0, 1]), float(bounds[1, 1]))

    heights, dx, _ = rasterize_mesh_heightfield(
      verts, faces, x_bounds, y_bounds, horizontal_scale=horizontal_scale
    )
    z_min = float(heights.min())
    z_max = float(heights.max())
    elev_z = max(z_max - z_min, _MIN_HFIELD_HEIGHT)
    base_z = max(elev_z, 0.5)
    normalized = np.clip((heights - z_min) / elev_z, 0.0, 1.0)

    half_x = max(0.5 * (x_bounds[1] - x_bounds[0]), _MIN_HFIELD_HEIGHT)
    half_y = max(0.5 * (y_bounds[1] - y_bounds[0]), _MIN_HFIELD_HEIGHT)
    cx = 0.5 * (x_bounds[0] + x_bounds[1])
    cy = 0.5 * (y_bounds[0] + y_bounds[1])

    hfield = spec.add_hfield(
      name=f"mesh_hfield_{uid}",
      size=[half_x, half_y, elev_z, base_z],
      nrow=heights.shape[0],
      ncol=heights.shape[1],
      userdata=normalized.flatten().astype(np.float32).tolist(),
    )
    hf_geom = body.add_geom(
      name=f"{geom_name}_collision" if geom_name else f"mesh_hfield_col_{uid}",
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=hfield.name,
      pos=(cx, cy, z_min),
      group=3,
      contype=1,
      conaffinity=1,
      rgba=_COLLISION_HFIELD_RGBA,
    )
    geometries.append(TerrainGeometry(geom=hf_geom, hfield=hfield))
    return geometries, heights, dx

  if collision_mode == "parts":
    sub_meshes: list[trimesh.Trimesh] = []
    for part in parts:
      split_parts = part.split(only_watertight=False)
      for sp in split_parts:
        if isinstance(sp, trimesh.Trimesh) and len(sp.faces) > 0:
          sub_meshes.append(sp)
    if not sub_meshes:
      sub_meshes = [combined]

    for idx, sub in enumerate(sub_meshes):
      sub_v, sub_f = _ensure_volumetric_mesh(
        np.asarray(sub.vertices, dtype=np.float64),
        np.asarray(sub.faces, dtype=np.int32),
      )
      m_asset = spec.add_mesh(
        name=f"mesh_terrain_{uid}_{idx}",
        uservert=sub_v.reshape(-1).tolist(),
        userface=sub_f.reshape(-1).tolist(),
      )
      part_name: str | None = None
      if geom_name is not None:
        part_name = geom_name if len(sub_meshes) == 1 else f"{geom_name}_part_{idx}"
      g = body.add_geom(
        name=part_name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=m_asset.name,
        group=0,
        contype=1,
        conaffinity=1,
        rgba=rgba,
      )
      geometries.append(TerrainGeometry(geom=g, color=rgba))
    return geometries, None, horizontal_scale

  if collision_mode == "convex":
    conv_v, conv_f = _ensure_volumetric_mesh(verts, faces)
    mesh_asset = spec.add_mesh(
      name=f"mesh_terrain_{uid}",
      uservert=conv_v.reshape(-1).tolist(),
      userface=conv_f.reshape(-1).tolist(),
    )
    g = body.add_geom(
      name=geom_name,
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_asset.name,
      group=0,
      contype=1,
      conaffinity=1,
      rgba=rgba,
    )
    geometries.append(TerrainGeometry(geom=g, color=rgba))
    return geometries, None, horizontal_scale

  raise ValueError(
    f"Invalid collision_mode '{collision_mode}'. "
    "Expected 'heightfield', 'parts', or 'convex'."
  )


@dataclass(kw_only=True)
class MeshTerrainCfg(SubTerrainCfg):
  """Sub-terrain patch imported from an external triangle mesh file."""

  mesh_path: str | Path
  """Path to the mesh file (e.g. ``.obj``, ``.stl``, ``.ply``, ``.glb``)."""
  scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
  """Per-axis ``(sx, sy, sz)`` scale factors applied to mesh vertices."""
  z_scale_range: tuple[float, float] | None = None
  """Optional difficulty-interpolated Z scale ``(min_sz, max_sz)`` overriding
  ``scale[2]`` across curriculum rows."""
  collision_mode: MeshCollisionMode = "heightfield"
  """Collision representation for MuJoCo:

  - ``"heightfield"``: Visual triangle mesh (group 0, non-colliding, raycastable)
    paired with a concave rasterized heightfield collider (group 3). Recommended
    for 2.5D terrain scans, hills, valleys, and stairs in a single welded mesh.
  - ``"parts"``: Splits disconnected mesh shells/components into separate
    ``mjGEOM_MESH`` colliders (convex hull per component). Recommended for
    multi-part CAD scenes, overhangs, or CoACD/V-HACD convex decompositions.
  - ``"convex"``: Single ``mjGEOM_MESH`` collider using MuJoCo's convex hull.
  """
  horizontal_scale: float = 0.05
  """Grid spacing (meters) used for heightfield collision rasterization and
  flat-patch detection."""
  center_xy: bool = True
  """If True, centers the mesh XY bounding box at ``(size[0] / 2, size[1] / 2)``."""
  align_min_z: bool = False
  """If True, shifts mesh vertices along Z so the minimum vertex Z is 0.0."""
  origin: tuple[float, float, float] | None = None
  """Explicit local spawn origin ``(x, y, z)``. When ``None``, defaults to the
  sub-terrain center ``(size[0] / 2, size[1] / 2)`` at the mesh top surface Z."""
  rgba: tuple[float, float, float, float] = _DEFAULT_MESH_RGBA
  """Default RGBA color applied to the mesh geometry."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    sz = self.scale[2]
    if self.z_scale_range is not None:
      lo, hi = self.z_scale_range
      sz = lo + difficulty * (hi - lo)
      # Guard against zero Z scale if user sets z_scale_range=(0.0, ...).
      sz = max(sz, _MIN_HFIELD_HEIGHT)
    eff_scale = (self.scale[0], self.scale[1], sz)

    center_target = (self.size[0] / 2.0, self.size[1] / 2.0) if self.center_xy else None
    parts, combined = load_terrain_meshes(
      self.mesh_path,
      scale=eff_scale,
      target_xy_center=center_target,
      align_min_z=self.align_min_z,
    )

    xy_bounds = ((0.0, self.size[0]), (0.0, self.size[1]))
    geometries, heights, dx = build_mesh_terrain_geometries(
      spec,
      body,
      parts,
      combined,
      collision_mode=self.collision_mode,
      horizontal_scale=self.horizontal_scale,
      xy_bounds=xy_bounds,
      rgba=self.rgba,
    )

    verts = np.asarray(combined.vertices, dtype=np.float64)
    faces = np.asarray(combined.faces, dtype=np.int32)

    if self.origin is not None:
      spawn_origin = np.asarray(self.origin, dtype=np.float64)
    else:
      cx, cy = self.size[0] / 2.0, self.size[1] / 2.0
      top_z = query_mesh_top_z(
        verts,
        faces,
        np.array([[cx, cy]], dtype=np.float64),
        fallback_z=float(np.max(verts[:, 2])),
      )[0]
      spawn_origin = np.array([cx, cy, top_z], dtype=np.float64)

    flat_patches: dict[str, np.ndarray] | None = None
    if self.flat_patch_sampling is not None:
      if heights is None:
        heights, dx, _ = rasterize_mesh_heightfield(
          verts,
          faces,
          xy_bounds[0],
          xy_bounds[1],
          horizontal_scale=self.horizontal_scale,
        )
      z_min = float(heights.min())
      rel_heights = heights - z_min
      flat_patches = {}
      for name, patch_cfg in self.flat_patch_sampling.items():
        flat_patches[name] = find_flat_patches_from_heightfield(
          heights=rel_heights,
          horizontal_scale=dx,
          z_offset=z_min,
          cfg=patch_cfg,
          rng=rng,
        )

    return TerrainOutput(
      origin=spawn_origin,
      geometries=geometries,
      flat_patches=flat_patches,
    )
