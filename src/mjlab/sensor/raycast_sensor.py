"""Raycast sensor for terrain and obstacle detection.

Provides :class:`RayCastSensor` and :class:`RayCastSensorCfg` for BVH-accelerated
raycasting with grid, pinhole camera, and ring patterns. Supports multi-frame
attachment, configurable ray alignment, and geom group filtering.

See :doc:`/sensors/raycast_sensor` for usage guide and examples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mujoco_warp import rays

from mjlab.entity import Entity
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.sensor import Sensor, SensorCfg
from mjlab.utils.lab_api.math import quat_from_matrix

if TYPE_CHECKING:
  from mjlab.sensor.sensor_context import SensorContext
  from mjlab.viewer.debug_visualizer import DebugVisualizer

RayAlignment = Literal["base", "yaw", "world"]

# NOTE: Not publicly exposed by mujoco_warp.
_vec6 = wp.types.vector(length=6, dtype=float)
_ALL_GROUPS = _vec6(-1, -1, -1, -1, -1, -1)


def _geom_groups_to_vec6(groups: tuple[int, ...] | None):  # -> _vec6
  """Convert geom group tuple to mujoco_warp vec6 format.

  In the vec6 format, -1 means include and 0 means exclude.
  """
  if groups is None:
    return _ALL_GROUPS
  out = [0, 0, 0, 0, 0, 0]
  for g in groups:
    if 0 <= g <= 5:
      out[g] = -1
  return _vec6(*out)


@dataclass
class GridPatternCfg:
  """Grid pattern - parallel rays in a 2D grid."""

  size: tuple[float, float] = (1.0, 1.0)
  """Grid size (length, width) in meters."""

  resolution: float = 0.1
  """Spacing between rays in meters."""

  direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
  """Ray direction in frame-local coordinates."""

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray pattern.

    Args:
      mj_model: MuJoCo model (unused for grid pattern).
      device: Device for tensor operations.

    Returns:
      Tuple of (local_offsets [N, 3], local_directions [N, 3]).
    """
    del mj_model  # Unused for grid pattern
    size_x, size_y = self.size
    res = self.resolution

    x = torch.arange(
      -size_x / 2, size_x / 2 + res * 0.5, res, device=device, dtype=torch.float32
    )
    y = torch.arange(
      -size_y / 2, size_y / 2 + res * 0.5, res, device=device, dtype=torch.float32
    )
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

    num_rays = grid_x.numel()
    local_offsets = torch.zeros((num_rays, 3), device=device, dtype=torch.float32)
    local_offsets[:, 0] = grid_x.flatten()
    local_offsets[:, 1] = grid_y.flatten()

    # All rays share the same direction for grid pattern.
    direction = torch.tensor(self.direction, device=device, dtype=torch.float32)
    direction = direction / direction.norm()
    local_directions = direction.unsqueeze(0).expand(num_rays, 3).clone()

    return local_offsets, local_directions


@dataclass
class PinholeCameraPatternCfg:
  """Pinhole camera pattern - rays diverging from origin like a camera.

  Can be configured with explicit parameters (width, height, fovy) or created
  via factory methods like from_mujoco_camera() or from_intrinsic_matrix().
  """

  width: int = 16
  """Image width in pixels."""

  height: int = 12
  """Image height in pixels."""

  fovy: float = 45.0
  """Vertical field of view in degrees (matches MuJoCo convention)."""

  _camera_name: str | None = field(default=None, repr=False)
  """Internal: MuJoCo camera name for deferred parameter resolution."""

  @classmethod
  def from_mujoco_camera(cls, camera_name: str) -> PinholeCameraPatternCfg:
    """Create config that references a MuJoCo camera.

    Camera parameters (resolution, FOV) are resolved at runtime from the model.

    Args:
      camera_name: Name of the MuJoCo camera to reference.

    Returns:
      Config that will resolve parameters from the MuJoCo camera.
    """
    # Placeholder values; actual values resolved in generate_rays().
    return cls(width=0, height=0, fovy=0.0, _camera_name=camera_name)

  @classmethod
  def from_intrinsic_matrix(
    cls, intrinsic_matrix: list[float], width: int, height: int
  ) -> PinholeCameraPatternCfg:
    """Create from 3x3 intrinsic matrix [fx, 0, cx, 0, fy, cy, 0, 0, 1].

    Args:
      intrinsic_matrix: Flattened 3x3 intrinsic matrix.
      width: Image width in pixels.
      height: Image height in pixels.

    Returns:
      Config with fovy computed from the intrinsic matrix.
    """
    fy = intrinsic_matrix[4]  # fy is at position [1,1] in the matrix
    fovy = 2 * math.atan(height / (2 * fy)) * 180 / math.pi
    return cls(width=width, height=height, fovy=fovy)

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray pattern.

    Args:
      mj_model: MuJoCo model (required if using from_mujoco_camera).
      device: Device for tensor operations.

    Returns:
      Tuple of (local_offsets [N, 3], local_directions [N, 3]).
    """
    # Resolve camera parameters.
    if self._camera_name is not None:
      if mj_model is None:
        raise ValueError("MuJoCo model required when using from_mujoco_camera()")
      # Get parameters from MuJoCo camera.
      cam_id = mj_model.camera(self._camera_name).id
      width, height = mj_model.cam_resolution[cam_id]

      # MuJoCo has two camera modes:
      # 1. fovy mode: sensorsize is zero, use cam_fovy directly
      # 2. Physical sensor mode: sensorsize > 0, compute from focal/sensorsize
      sensorsize = mj_model.cam_sensorsize[cam_id]
      if sensorsize[0] > 0 and sensorsize[1] > 0:
        # Physical sensor model.
        intrinsic = mj_model.cam_intrinsic[cam_id]  # [fx, fy, cx, cy]
        focal = intrinsic[:2]  # [fx, fy]
        h_fov_rad = 2 * math.atan(sensorsize[0] / (2 * focal[0]))
        v_fov_rad = 2 * math.atan(sensorsize[1] / (2 * focal[1]))
      else:
        # Read vertical FOV directly from MuJoCo.
        v_fov_rad = math.radians(mj_model.cam_fovy[cam_id])
        aspect = width / height
        h_fov_rad = 2 * math.atan(math.tan(v_fov_rad / 2) * aspect)
    else:
      # Use explicit parameters.
      width = self.width
      height = self.height
      v_fov_rad = math.radians(self.fovy)
      aspect = width / height
      h_fov_rad = 2 * math.atan(math.tan(v_fov_rad / 2) * aspect)

    # Create normalized pixel coordinates [-1, 1].
    u = torch.linspace(-1, 1, width, device=device, dtype=torch.float32)
    v = torch.linspace(-1, 1, height, device=device, dtype=torch.float32)
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")

    # Convert to ray directions (MuJoCo camera: -Z forward, +X right, +Y down).
    ray_x = grid_u.flatten() * math.tan(h_fov_rad / 2)
    ray_y = grid_v.flatten() * math.tan(v_fov_rad / 2)
    ray_z = -torch.ones_like(ray_x)  # Negative Z for MuJoCo camera forward

    num_rays = width * height
    local_offsets = torch.zeros((num_rays, 3), device=device)
    local_directions = torch.stack([ray_x, ray_y, ray_z], dim=1)
    local_directions = local_directions / local_directions.norm(dim=1, keepdim=True)

    return local_offsets, local_directions


@dataclass
class RingPatternCfg:
  """Ring pattern - rays in concentric rings around the origin.

  Useful for per-site height sensing where you want to sample the
  terrain around each attachment frame.
  """

  @dataclass
  class Ring:
    """A single ring in the pattern."""

    radius: float
    """Radius of this ring in meters."""

    num_samples: int
    """Number of evenly spaced sample points on this ring."""

  rings: tuple[Ring, ...]
  """Ring definitions. Multiple rings create a concentric pattern."""

  include_center: bool = True
  """Whether to include a ray at the center (zero offset)."""

  direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
  """Ray direction in frame-local coordinates."""

  @classmethod
  def single_ring(
    cls,
    radius: float = 0.1,
    num_samples: int = 8,
    include_center: bool = True,
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
  ) -> RingPatternCfg:
    """Create a single-ring pattern.

    Args:
      radius: Ring radius in meters.
      num_samples: Number of evenly spaced points on the ring.
      include_center: Whether to include a center ray.
      direction: Ray direction in frame-local coordinates.

    Returns:
      RingPatternCfg with one ring.
    """
    return cls(
      rings=(cls.Ring(radius, num_samples),),
      include_center=include_center,
      direction=direction,
    )

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray pattern.

    Args:
      mj_model: MuJoCo model (unused for ring pattern).
      device: Device for tensor operations.

    Returns:
      Tuple of (local_offsets [N, 3], local_directions [N, 3]).
    """
    del mj_model
    offsets: list[torch.Tensor] = []

    if self.include_center:
      offsets.append(torch.zeros(3, device=device, dtype=torch.float32))

    for ring in self.rings:
      for i in range(ring.num_samples):
        angle = 2.0 * math.pi * i / ring.num_samples
        offsets.append(
          torch.tensor(
            [
              ring.radius * math.cos(angle),
              ring.radius * math.sin(angle),
              0.0,
            ],
            device=device,
            dtype=torch.float32,
          )
        )

    local_offsets = torch.stack(offsets)  # [N, 3]
    num_rays = local_offsets.shape[0]

    direction = torch.tensor(self.direction, device=device, dtype=torch.float32)
    direction = direction / direction.norm()
    local_directions = direction.unsqueeze(0).expand(num_rays, 3).clone()

    return local_offsets, local_directions


PatternCfg = GridPatternCfg | PinholeCameraPatternCfg | RingPatternCfg


@dataclass
class RayCastData:
  """Raycast sensor output data.

  Note:
    Fields are views into GPU buffers and are valid until the next
    ``sense()`` call.
  """

  distances: torch.Tensor
  """[B, N] Distance to hit point. -1 if no hit.

  N = num_frames * num_rays_per_frame.
  """

  normals_w: torch.Tensor
  """[B, N, 3] Surface normal at hit point (world frame). Zero if no hit."""

  hit_pos_w: torch.Tensor
  """[B, N, 3] Hit position in world frame. Ray origin if no hit."""

  pos_w: torch.Tensor
  """[B, 3] First frame position in world coordinates."""

  quat_w: torch.Tensor
  """[B, 4] First frame orientation quaternion (w, x, y, z)."""

  frame_pos_w: torch.Tensor
  """[B, F, 3] All frame positions in world coordinates."""

  frame_quat_w: torch.Tensor
  """[B, F, 4] All frame orientations (w, x, y, z)."""


@dataclass
class RayCastSensorCfg(SensorCfg):
  """Raycast sensor configuration.

  Supports multiple ray patterns (grid, pinhole camera, ring) and alignment modes.
  """

  @dataclass
  class VizCfg:
    """Visualization settings for debug rendering."""

    hit_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.8)
    """RGBA color for rays that hit a surface."""

    miss_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.4)
    """RGBA color for rays that miss."""

    hit_sphere_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)
    """RGBA color for spheres drawn at hit points."""

    hit_sphere_radius: float = 0.5
    """Radius of spheres drawn at hit points (multiplier of meansize)."""

    show_rays: bool = False
    """Whether to draw ray arrows."""

    show_normals: bool = False
    """Whether to draw surface normals at hit points."""

    normal_color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)
    """RGBA color for surface normal arrows."""

    normal_length: float = 5.0
    """Length of surface normal arrows (multiplier of meansize)."""

  frame: ObjRef | tuple[ObjRef, ...]
  """Body, site, or geom to attach rays to.

  Pass a single ``ObjRef`` for one frame, or a tuple for multi-frame
  sensing (e.g. per-foot height sensors). Each frame's parent body is
  excluded independently when ``exclude_parent_body`` is True.
  """

  pattern: PatternCfg = field(default_factory=GridPatternCfg)
  """Ray pattern configuration. Defaults to GridPatternCfg."""

  ray_alignment: RayAlignment = "base"
  """How rays are oriented relative to the frame. Controls direction only;
  the ray origin is always the physical frame position (``site_xpos`` /
  ``geom_xpos`` / ``body_xpos``).

  - "base": Full rotation (default). Rays rotate with the body.
  - "yaw": Yaw only, ignores pitch/roll (good for height maps).
  - "world": Fixed in world frame, rays always point the same direction.
  """

  max_distance: float = 10.0
  """Maximum ray distance. Rays beyond this report -1."""

  exclude_parent_body: bool = True
  """Exclude parent body from ray intersection tests."""

  include_geom_groups: tuple[int, ...] | None = (0, 1, 2)
  """Geom groups (0-5) to include in raycasting.

  Defaults to (0, 1, 2). Set to None to include all groups.
  """

  debug_vis: bool = False
  """Enable debug visualization."""

  viz: VizCfg = field(default_factory=VizCfg)
  """Visualization settings."""

  def build(self) -> RayCastSensor:
    return RayCastSensor(self)


class RayCastSensor(Sensor[RayCastData]):
  """Raycast sensor for terrain and obstacle detection."""

  requires_sensor_context = True

  def __init__(self, cfg: RayCastSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self._data: mjwarp.Data | None = None
    self._model: mjwarp.Model | None = None
    self._mj_model: mujoco.MjModel | None = None
    self._device: str | None = None
    self._wp_device: wp.context.Device | None = None

    # Per-frame info: list of (frame_type, obj_id, body_id).
    self._frame_infos: list[tuple[Literal["body", "site", "geom"], int, int]] = []
    self._num_frames: int = 0
    self._num_rays_per_frame: int = 0

    self._local_offsets: torch.Tensor | None = None
    self._local_directions: torch.Tensor | None = None  # [rays_per_frame, 3]
    self._num_rays: int = 0

    self._ray_pnt: wp.array | None = None
    self._ray_vec: wp.array | None = None
    self._ray_dist: wp.array | None = None
    self._ray_geomid: wp.array | None = None
    self._ray_normal: wp.array | None = None
    self._ray_bodyexclude: wp.array | None = None
    self._geomgroup = _vec6(-1, -1, -1, -1, -1, -1)

    self._distances: torch.Tensor | None = None
    self._normals_w: torch.Tensor | None = None
    self._hit_pos_w: torch.Tensor | None = None
    self._pos_w: torch.Tensor | None = None
    self._quat_w: torch.Tensor | None = None
    self._frame_pos_w: torch.Tensor | None = None
    self._frame_quat_w: torch.Tensor | None = None

    self._cached_world_origins: torch.Tensor | None = None
    self._cached_world_rays: torch.Tensor | None = None
    self._cached_frame_pos: torch.Tensor | None = None
    self._cached_frame_mat: torch.Tensor | None = None

    self._debug_vis_enabled: bool = True
    self._ctx: SensorContext | None = None

  def edit_spec(
    self,
    scene_spec: mujoco.MjSpec,
    entities: dict[str, Entity],
  ) -> None:
    del scene_spec, entities

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    self._data = data
    self._model = model
    self._mj_model = mj_model
    self._device = device
    self._wp_device = wp.get_device(device)
    num_envs = data.nworld

    # Normalize frame to tuple.
    frames = self.cfg.frame
    if isinstance(frames, ObjRef):
      frames = (frames,)

    # Resolve per-frame IDs.
    self._frame_infos = []
    for frame in frames:
      frame_name = frame.prefixed_name()
      info: tuple[Literal["body", "site", "geom"], int, int]
      if frame.type == "body":
        bid = mj_model.body(frame_name).id
        info = ("body", bid, bid)
      elif frame.type == "site":
        sid = mj_model.site(frame_name).id
        info = ("site", sid, int(mj_model.site_bodyid[sid]))
      elif frame.type == "geom":
        gid = mj_model.geom(frame_name).id
        info = ("geom", gid, int(mj_model.geom_bodyid[gid]))
      else:
        raise ValueError(
          f"RayCastSensor frame must be 'body', 'site', or 'geom', got '{frame.type}'"
        )
      self._frame_infos.append(info)
    self._num_frames = len(self._frame_infos)

    # Generate ray pattern.
    pattern = self.cfg.pattern
    self._local_offsets, self._local_directions = pattern.generate_rays(
      mj_model, device
    )
    self._num_rays_per_frame = self._local_offsets.shape[0]
    self._num_rays = self._num_frames * self._num_rays_per_frame

    self._ray_pnt = wp.zeros((num_envs, self._num_rays), dtype=wp.vec3, device=device)
    self._ray_vec = wp.zeros((num_envs, self._num_rays), dtype=wp.vec3, device=device)
    self._ray_dist = wp.zeros((num_envs, self._num_rays), dtype=float, device=device)
    self._ray_geomid = wp.zeros((num_envs, self._num_rays), dtype=int, device=device)
    self._ray_normal = wp.zeros(
      (num_envs, self._num_rays), dtype=wp.vec3, device=device
    )

    # Body exclusion: each frame's body_id repeated N times.
    if self.cfg.exclude_parent_body:
      body_excludes: list[int] = []
      for _, _, body_id in self._frame_infos:
        body_excludes.extend([body_id] * self._num_rays_per_frame)
    else:
      body_excludes = [-1] * self._num_rays
    self._ray_bodyexclude = wp.array(body_excludes, dtype=int, device=device)

    self._geomgroup = _geom_groups_to_vec6(self.cfg.include_geom_groups)

    # Pre-allocate output tensors so shape inference works before
    # the first sense() call.
    F = self._num_frames
    self._distances = torch.zeros(num_envs, self._num_rays, device=device)
    self._normals_w = torch.zeros(num_envs, self._num_rays, 3, device=device)
    self._hit_pos_w = torch.zeros(num_envs, self._num_rays, 3, device=device)
    self._pos_w = torch.zeros(num_envs, 3, device=device)
    self._quat_w = torch.zeros(num_envs, 4, device=device)
    self._frame_pos_w = torch.zeros(num_envs, F, 3, device=device)
    self._frame_quat_w = torch.zeros(num_envs, F, 4, device=device)

    assert self._wp_device is not None

  @property
  def include_geom_groups(self) -> tuple[int, ...] | None:
    return self.cfg.include_geom_groups

  def set_context(self, ctx: SensorContext) -> None:
    """Wire this sensor to a SensorContext for BVH-accelerated raycasting."""
    self._ctx = ctx

  def _compute_data(self) -> RayCastData:
    if self._ctx is None:
      raise RuntimeError(
        "RayCastSensor requires a SensorContext. "
        "Ensure the sensor is part of a scene with "
        "sim.sense() calls."
      )
    assert self._distances is not None and self._normals_w is not None
    assert self._hit_pos_w is not None
    assert self._pos_w is not None and self._quat_w is not None
    assert self._frame_pos_w is not None and self._frame_quat_w is not None
    return RayCastData(
      distances=self._distances,
      normals_w=self._normals_w,
      hit_pos_w=self._hit_pos_w,
      pos_w=self._pos_w,
      quat_w=self._quat_w,
      frame_pos_w=self._frame_pos_w,
      frame_quat_w=self._frame_quat_w,
    )

  @property
  def num_rays(self) -> int:
    """Total number of rays (num_frames * num_rays_per_frame)."""
    return self._num_rays

  @property
  def num_frames(self) -> int:
    """Number of attachment frames."""
    return self._num_frames

  @property
  def num_rays_per_frame(self) -> int:
    """Number of rays per attachment frame."""
    return self._num_rays_per_frame

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self.cfg.debug_vis or not self._debug_vis_enabled:
      return
    assert self._data is not None
    assert self._local_offsets is not None
    assert self._local_directions is not None
    assert self._cached_frame_pos is not None
    assert self._cached_frame_mat is not None

    data = self.data
    env_indices = list(visualizer.get_env_indices(data.distances.shape[0]))
    if not env_indices:
      return

    F = self._num_frames
    N = self._num_rays_per_frame

    # Use cached frame poses [B, F, 3] / [B, F, 3, 3].
    frame_pos = self._cached_frame_pos[env_indices]  # [K, F, 3]
    frame_mat = self._cached_frame_mat[env_indices]  # [K, F, 3, 3]

    K_envs = len(env_indices)
    # Compute alignment rotation for all frames.
    rot_mats = (
      self._compute_alignment_rotation(frame_mat.view(K_envs * F, 3, 3))
      .view(K_envs, F, 3, 3)
      .cpu()
      .numpy()
    )
    origins = frame_pos.cpu().numpy()
    offsets = self._local_offsets.cpu().numpy()
    directions = self._local_directions.cpu().numpy()
    hit_positions = data.hit_pos_w[env_indices].cpu().numpy()
    distances = data.distances[env_indices].cpu().numpy()
    normals = data.normals_w[env_indices].cpu().numpy()

    meansize = visualizer.meansize
    ray_width = 0.1 * meansize
    sphere_radius = self.cfg.viz.hit_sphere_radius * meansize
    normal_length = self.cfg.viz.normal_length * meansize
    normal_width = 0.1 * meansize
    miss_extent = min(0.5, self.cfg.max_distance * 0.05)
    name = self.cfg.name

    for k in range(K_envs):
      for f in range(F):
        rot = rot_mats[k, f]
        for i in range(N):
          ray_idx = f * N + i
          origin = origins[k, f] + rot @ offsets[i]
          hit = distances[k, ray_idx] >= 0

          if hit:
            end = hit_positions[k, ray_idx]
            color = self.cfg.viz.hit_color
          else:
            end = origin + rot @ directions[i] * miss_extent
            color = self.cfg.viz.miss_color

          if self.cfg.viz.show_rays:
            visualizer.add_arrow(
              start=origin,
              end=end,
              color=color,
              width=ray_width,
              label=f"{name}_ray_{ray_idx}",
            )

          if hit:
            visualizer.add_sphere(
              center=end,
              radius=sphere_radius,
              color=self.cfg.viz.hit_sphere_color,
              label=f"{name}_hit_{ray_idx}",
            )
            if self.cfg.viz.show_normals:
              normal_end = end + normals[k, ray_idx] * normal_length
              visualizer.add_arrow(
                start=end,
                end=normal_end,
                color=self.cfg.viz.normal_color,
                width=normal_width,
                label=f"{name}_normal_{ray_idx}",
              )

  # Private methods.

  def prepare_rays(self) -> None:
    """PRE-GRAPH: Transform local rays to world frame.

    Reads body/site/geom poses via PyTorch and writes world-frame ray
    origins and directions into Warp arrays. Caches the frame pose and
    world-frame tensors for postprocess_rays().
    """
    assert self._data is not None and self._model is not None
    assert self._local_offsets is not None and self._local_directions is not None

    # Gather per-frame poses: [B, F, 3] and [B, F, 3, 3].
    # Position is always the physical world position. Alignment only
    # affects ray directions (applied to frame_mat below).
    pos_list: list[torch.Tensor] = []
    mat_list: list[torch.Tensor] = []
    for frame_type, obj_id, _ in self._frame_infos:
      if frame_type == "body":
        pos_list.append(self._data.xpos[:, obj_id])
        mat_list.append(self._data.xmat[:, obj_id].view(-1, 3, 3))
      elif frame_type == "site":
        pos_list.append(self._data.site_xpos[:, obj_id])
        mat_list.append(self._data.site_xmat[:, obj_id].view(-1, 3, 3))
      else:  # geom
        pos_list.append(self._data.geom_xpos[:, obj_id])
        mat_list.append(self._data.geom_xmat[:, obj_id].view(-1, 3, 3))

    frame_pos = torch.stack(pos_list, dim=1)  # [B, F, 3]
    frame_mat = torch.stack(mat_list, dim=1)  # [B, F, 3, 3]

    B, F = frame_pos.shape[:2]
    N = self._num_rays_per_frame

    # Compute alignment rotation for all frames at once.
    rot_mat = self._compute_alignment_rotation(frame_mat.reshape(B * F, 3, 3)).reshape(
      B, F, 3, 3
    )

    # Compute world offsets and directions: [B, F, N, 3].
    world_offsets = torch.einsum("bfij,nj->bfni", rot_mat, self._local_offsets)
    world_origins = frame_pos[:, :, None, :] + world_offsets
    world_rays = torch.einsum("bfij,nj->bfni", rot_mat, self._local_directions)

    # Flatten to [B, F*N, 3] for raycasting.
    world_origins_flat = world_origins.reshape(B, F * N, 3)
    world_rays_flat = world_rays.reshape(B, F * N, 3)

    assert self._ray_pnt is not None and self._ray_vec is not None
    pnt_torch = wp.to_torch(self._ray_pnt).view(B, self._num_rays, 3)
    vec_torch = wp.to_torch(self._ray_vec).view(B, self._num_rays, 3)
    pnt_torch.copy_(world_origins_flat)
    vec_torch.copy_(world_rays_flat)

    # Cache for postprocess_rays() and debug_vis().
    self._cached_world_origins = world_origins_flat
    self._cached_world_rays = world_rays_flat
    self._cached_frame_pos = frame_pos  # [B, F, 3]
    self._cached_frame_mat = frame_mat  # [B, F, 3, 3]

  def raycast_kernel(self, rc: mjwarp.RenderContext) -> None:
    """IN-GRAPH: Execute BVH-accelerated raycast kernel."""
    rays(
      m=self._model.struct,  # type: ignore[attr-defined]
      d=self._data.struct,  # type: ignore[attr-defined]
      pnt=self._ray_pnt,
      vec=self._ray_vec,
      geomgroup=self._geomgroup,  # pyright: ignore[reportArgumentType]
      flg_static=True,
      bodyexclude=self._ray_bodyexclude,
      dist=self._ray_dist,
      geomid=self._ray_geomid,
      normal=self._ray_normal,
      rc=rc,
    )

  def postprocess_rays(self) -> None:
    """POST-GRAPH: Convert Warp outputs to PyTorch, compute hit positions."""
    assert self._cached_world_origins is not None
    assert self._cached_world_rays is not None
    assert self._cached_frame_pos is not None
    assert self._cached_frame_mat is not None

    B = self._cached_frame_pos.shape[0]
    F = self._num_frames

    assert self._ray_dist is not None and self._ray_normal is not None
    distances = wp.to_torch(self._ray_dist)
    normals_w = wp.to_torch(self._ray_normal).view(B, self._num_rays, 3)
    distances[distances > self.cfg.max_distance] = -1.0

    hit_mask = distances >= 0
    hit_pos_w = self._cached_world_origins.clone()
    hit_pos_w[hit_mask] = self._cached_world_origins[
      hit_mask
    ] + self._cached_world_rays[hit_mask] * distances[hit_mask].unsqueeze(-1)
    self._hit_pos_w = hit_pos_w

    # Zero out normals for misses.
    normals_w[~hit_mask] = 0.0
    self._distances = distances
    self._normals_w = normals_w

    # All frames: [B, F, 3] / [B, F, 4].
    self._frame_pos_w = self._cached_frame_pos
    self._frame_quat_w = quat_from_matrix(
      self._cached_frame_mat.reshape(B * F, 3, 3)
    ).reshape(B, F, 4)

    # First frame for backward compat: [B, 3] / [B, 4].
    assert self._frame_pos_w is not None and self._frame_quat_w is not None
    self._pos_w = self._frame_pos_w[:, 0]
    self._quat_w = self._frame_quat_w[:, 0]

  def _compute_alignment_rotation(self, frame_mat: torch.Tensor) -> torch.Tensor:
    """Compute rotation matrix based on ray_alignment setting."""
    if self.cfg.ray_alignment == "base":
      # Full rotation.
      return frame_mat
    elif self.cfg.ray_alignment == "yaw":
      # Extract yaw only, zero out pitch/roll.
      return self._extract_yaw_rotation(frame_mat)
    elif self.cfg.ray_alignment == "world":
      # Identity rotation (world-aligned).
      num_envs = frame_mat.shape[0]
      return (
        torch.eye(3, device=frame_mat.device, dtype=frame_mat.dtype)
        .unsqueeze(0)
        .expand(num_envs, -1, -1)
      )
    else:
      raise ValueError(f"Unknown ray_alignment: {self.cfg.ray_alignment}")

  def _extract_yaw_rotation(self, rot_mat: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only rotation matrix (rotation around Z axis).

    Handles the singularity at ±90° pitch by falling back to the Y-axis
    when the X-axis projection onto the XY plane is too small.
    """
    batch_size = rot_mat.shape[0]
    device = rot_mat.device
    dtype = rot_mat.dtype

    # Project X-axis onto XY plane.
    x_axis = rot_mat[:, :, 0]  # First column [B, 3]
    x_proj = x_axis.clone()
    x_proj[:, 2] = 0  # Zero out Z component
    x_norm = x_proj.norm(dim=1)  # [B]

    # Check for singularity (X-axis nearly vertical).
    threshold = 0.1
    singular = x_norm < threshold  # [B]

    # For singular cases, use Y-axis instead.
    if singular.any():
      y_axis = rot_mat[:, :, 1]  # Second column [B, 3]
      y_proj = y_axis.clone()
      y_proj[:, 2] = 0
      y_norm = y_proj.norm(dim=1).clamp(min=1e-6)
      y_proj = y_proj / y_norm.unsqueeze(-1)
      # Y-axis points left; rotate -90° around Z to get forward direction.
      # [y_x, y_y] -> [y_y, -y_x]
      x_from_y = torch.zeros_like(y_proj)
      x_from_y[:, 0] = y_proj[:, 1]
      x_from_y[:, 1] = -y_proj[:, 0]
      x_proj[singular] = x_from_y[singular]
      x_norm[singular] = 1.0  # Already normalized

    # Normalize X projection.
    x_norm = x_norm.clamp(min=1e-6)
    x_proj = x_proj / x_norm.unsqueeze(-1)

    # Build yaw-only rotation matrix.
    yaw_mat = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
    yaw_mat[:, 0, 0] = x_proj[:, 0]
    yaw_mat[:, 1, 0] = x_proj[:, 1]
    yaw_mat[:, 0, 1] = -x_proj[:, 1]
    yaw_mat[:, 1, 1] = x_proj[:, 0]
    yaw_mat[:, 2, 2] = 1
    return yaw_mat
