"""Tactile sensor: a grid of taxels on the surface of a collision geom.

Wraps MuJoCo's tactile sensor. Each taxel reports how deep a contacting object
reaches at that point, plus how fast the object slides past it. The taxels are
placed by projecting a regular grid onto the geom's collision surface (the
convex hull MuJoCo collides with, for meshes), so they lie exactly where
objects touch and the readings come back as images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass
class TactileData:
  """Tactile readings, laid out as images over the taxel grid."""

  depth: torch.Tensor
  """Penetration depth of the contacting object at each taxel [B, R, C] (m)."""

  slip: torch.Tensor
  """Tangential sliding speed along the pad's x and y axes [B, R, C, 2] (m/s)."""


@dataclass
class TactileSensorCfg(SensorCfg):
  """A taxel grid projected onto the surface of a collision geom.

  The grid lives in a pad frame given in the geom's body frame. Its +z axis
  points out of the surface: each taxel is cast from the pad's xy plane along
  -z and placed where it first hits the geom.
  """

  geom: ObjRef
  """Collision geom carrying the taxels. Must have type "geom"."""

  pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Pad center in the geom's body frame."""

  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  """Pad orientation in the geom's body frame (w, x, y, z)."""

  size: tuple[float, float] = (0.016, 0.010)
  """Grid extent along the pad's x and y axes (m)."""

  resolution: tuple[int, int] = (10, 7)
  """Number of taxels along the pad's x and y axes."""

  cutoff: float = 0.0
  """When positive, clamps every reading to this value."""

  debug_vis: bool = False
  """Enable debug visualization."""

  @dataclass
  class VizCfg:
    depth_range: float = 0.001
    """Depth (m) that maps to the hottest color."""

    idle_color: tuple[float, float, float, float] = (0.55, 0.55, 0.6, 0.35)
    """Color of taxels that are not in contact."""

  viz: VizCfg = field(default_factory=VizCfg)

  def __post_init__(self) -> None:
    if self.geom.type != "geom":
      raise ValueError(f"TactileSensorCfg.geom must be a geom, got {self.geom.type}")
    if min(self.resolution) < 2:
      raise ValueError(f"resolution must be at least 2x2, got {self.resolution}")

  @property
  def prefixed_name(self) -> str:
    if self.geom.entity is not None:
      return f"{self.geom.entity}/{self.name}"
    return self.name

  def build(self) -> TactileSensor:
    return TactileSensor(self)


class TactileSensor(Sensor[TactileData]):
  """Taxel grid sensor returning :class:`TactileData` views into sensordata."""

  def __init__(self, cfg: TactileSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self._name = cfg.prefixed_name
    self._taxel_name = f"{self._name}_taxels"
    self._debug_vis_enabled = True
    self._mjdata: mjwarp.Data | None = None
    self._view: TactileData | None = None
    self._taxel_geom_id = -1
    self._taxels_local: torch.Tensor | None = None

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del entities
    cfg = self.cfg
    geom_name = cfg.geom.prefixed_name()
    model = scene_spec.copy().compile()
    points, frames = _project_grid(model, model.geom(geom_name).id, cfg)

    rows, cols = cfg.resolution
    idx = np.arange(rows * cols).reshape(rows, cols)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    faces = np.stack([np.stack([a, b, c], -1), np.stack([d, c, b], -1)])
    scene_spec.add_mesh(
      name=self._taxel_name,
      uservert=points.ravel().tolist(),
      usernormal=frames.ravel().tolist(),
      userface=faces.ravel().tolist(),
      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    # The taxel geom only poses the taxels; the geom it covers makes the contacts.
    scene_spec.geom(geom_name).parent.add_geom(
      name=self._taxel_name,
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=self._taxel_name,
      contype=0,
      conaffinity=0,
      density=0,
      group=5,
      rgba=(0, 0, 0, 0),
    )
    scene_spec.add_sensor(
      name=self._name,
      type=mujoco.mjtSensor.mjSENS_TACTILE,
      objtype=mujoco.mjtObj.mjOBJ_MESH,
      objname=self._taxel_name,
      reftype=mujoco.mjtObj.mjOBJ_GEOM,
      refname=self._taxel_name,
      cutoff=cfg.cutoff if cfg.cutoff > 0 else None,
    )

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    del model
    self._mjdata = data
    sensor = mj_model.sensor(self._name)
    adr, dim = sensor.adr[0], sensor.dim[0]
    # MuJoCo stores the channels as contiguous blocks: [depth, slip_x, slip_y].
    channels = data.sensordata[:, adr : adr + dim].unflatten(
      1, (3, *self.cfg.resolution)
    )
    self._view = TactileData(
      depth=channels[:, 0], slip=channels[:, 1:].permute(0, 2, 3, 1)
    )
    self._taxel_geom_id = mj_model.geom(self._taxel_name).id
    mesh = mj_model.geom_dataid[self._taxel_geom_id]
    start = mj_model.mesh_vertadr[mesh]
    verts = mj_model.mesh_vert[start : start + mj_model.mesh_vertnum[mesh]]
    self._taxels_local = torch.tensor(verts, dtype=torch.float32, device=device)

  def _compute_data(self) -> TactileData:
    assert self._view is not None
    return self._view

  def taxel_pos_w(self) -> torch.Tensor:
    """World positions of the taxels [B, R * C, 3], in row-major grid order."""
    assert self._mjdata is not None and self._taxels_local is not None
    pos = self._mjdata.geom_xpos[:, self._taxel_geom_id]
    mat = self._mjdata.geom_xmat[:, self._taxel_geom_id].view(-1, 3, 3)
    return pos[:, None] + self._taxels_local @ mat.transpose(1, 2)

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self.cfg.debug_vis or not self._debug_vis_enabled:
      return
    depth = self.data.depth
    env_ids = list(visualizer.get_env_indices(depth.shape[0]))
    if not env_ids:
      return
    pos = self.taxel_pos_w()[env_ids].cpu().numpy()
    heat = (depth[env_ids].flatten(1) / self.cfg.viz.depth_range).clamp(0, 1)
    heat = heat.cpu().numpy()
    rows, cols = self.cfg.resolution
    radius = 0.3 * min(self.cfg.size[0] / rows, self.cfg.size[1] / cols)
    for k in range(len(env_ids)):
      for p, h in zip(pos[k], heat[k], strict=True):
        color = _heat_color(h) if h > 0 else self.cfg.viz.idle_color
        visualizer.add_sphere(p, radius * (1.0 + 0.6 * h), color)


def _heat_color(h: float) -> tuple[float, float, float, float]:
  """Yellow at light contact to deep red at full depth."""
  return (1.0 - 0.2 * h, 0.85 * (1.0 - h) + 0.05, 0.15 * (1.0 - h), 1.0)


def _project_grid(
  model: mujoco.MjModel, geom_id: int, cfg: TactileSensorCfg
) -> tuple[np.ndarray, np.ndarray]:
  """Cast the taxel grid onto the geom's collision surface, in its body frame.

  Returns taxel positions [R * C, 3] and per-taxel frames [R * C, 3, 3] whose
  rows are the surface normal, then the tangents along the pad's x and y.
  """
  pad = np.zeros(9)
  mujoco.mju_quat2Mat(pad, np.array(cfg.quat, dtype=np.float64))
  pad = pad.reshape(3, 3)
  rows, cols = cfg.resolution
  u = ((np.arange(rows) + 0.5) / rows - 0.5) * cfg.size[0]
  v = ((np.arange(cols) + 0.5) / cols - 0.5) * cfg.size[1]
  uv = np.stack(np.meshgrid(u, v, indexing="ij"), -1).reshape(-1, 2)

  geom_pos = model.geom_pos[geom_id]
  geom_mat = np.zeros(9)
  mujoco.mju_quat2Mat(geom_mat, model.geom_quat[geom_id])
  geom_mat = geom_mat.reshape(3, 3)
  # Start every ray outside the geom's bounding sphere.
  standoff = model.geom_rbound[geom_id] + np.linalg.norm(np.subtract(cfg.pos, geom_pos))
  origins = cfg.pos + uv @ pad[:, :2].T + standoff * pad[:, 2]
  direction = -pad[:, 2]

  geom_type = model.geom_type[geom_id]
  if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
    dist, normals = _ray_hull(model, geom_id, geom_pos, geom_mat, origins, direction)
  elif geom_type in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
    raise ValueError(f"Tactile sensors cannot cover {mujoco.mjtGeom(geom_type).name}")
  else:
    dist, normals = np.full(len(origins), -1.0), np.zeros((len(origins), 3))
    for i, origin in enumerate(origins):
      dist[i] = mujoco.mju_rayGeom(
        geom_pos,
        geom_mat.ravel(),
        model.geom_size[geom_id],
        origin,
        direction,
        geom_type,
        normals[i],
      )

  if np.any(dist < 0):
    raise ValueError(
      f"Tactile sensor '{cfg.name}': {int(np.sum(dist < 0))} of {len(dist)} taxels "
      f"miss geom '{cfg.geom.prefixed_name()}'. Shrink size or recenter the pad."
    )
  points = origins + dist[:, None] * direction
  tangent_x = pad[:, 0] - (normals @ pad[:, 0])[:, None] * normals
  tangent_x /= np.linalg.norm(tangent_x, axis=1, keepdims=True)
  tangent_y = np.cross(normals, tangent_x)
  return points, np.stack([normals, tangent_x, tangent_y], axis=1)


def _ray_hull(
  model: mujoco.MjModel,
  geom_id: int,
  geom_pos: np.ndarray,
  geom_mat: np.ndarray,
  origins: np.ndarray,
  direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Intersect rays with the convex hull MuJoCo collides the mesh geom against."""
  mesh = model.geom_dataid[geom_id]
  graphadr = model.mesh_graphadr[mesh]
  if graphadr < 0:
    raise ValueError(f"Mesh geom '{model.geom(geom_id).name}' has no convex hull.")
  graph = model.mesh_graph[graphadr:]
  nvert, nface = graph[0], graph[1]
  faces = graph[2 + 3 * nvert + 3 * nface :][: 3 * nface].reshape(-1, 3)
  start = model.mesh_vertadr[mesh]
  verts = model.mesh_vert[start : start + model.mesh_vertnum[mesh]].astype(np.float64)
  verts = verts @ geom_mat.T + geom_pos

  # Outward face planes n . x + c = 0.
  tri = verts[faces]
  normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  offsets = -np.einsum("fi,fi->f", normals, tri[:, 0])
  flip = normals @ verts.mean(0) + offsets > 0
  normals[flip], offsets[flip] = -normals[flip], -offsets[flip]

  # Clip each ray against every face plane; it enters at the last entering plane.
  denom = normals @ direction
  t = -(origins @ normals.T + offsets) / np.where(denom == 0, np.inf, denom)
  t_enter = np.where(denom < 0, t, -np.inf)
  entry = t_enter.argmax(1)
  t_in = t_enter.max(1)
  t_out = np.where(denom > 0, t, np.inf).min(1)
  hit = (t_in <= t_out) & (t_in >= 0)
  return np.where(hit, t_in, -1.0), normals[entry]
