"""Raycast sensor for terrain and obstacle detection.

Ray Patterns
------------

This module provides two ray pattern types for different use cases:

**Grid Pattern** - Parallel rays in a 2D grid::

    Camera at any height:
          ↓   ↓   ↓   ↓   ↓      ← All rays point same direction
          ↓   ↓   ↓   ↓   ↓
          ↓   ↓   ↓   ↓   ↓
        ──●───●───●───●───●──    ← Fixed spacing (e.g., 10cm apart)
             Ground

- Rays are parallel (all point in the same direction, e.g., -Z down)
- Spacing is defined in world units (meters)
- Height doesn't affect the hit pattern - same footprint regardless of altitude
- Good for: height maps, terrain scanning with consistent spatial sampling

**Pinhole Camera Pattern** - Diverging rays from a single point::

    Camera LOW:                    Camera HIGH:
             📷                            📷
            /|\\                          /  |  \\
           / | \\                        /   |   \\
          /  |  \\                      /    |    \\
        ─●───●───●─                 ───●─────●─────●───
        (small footprint)           (large footprint)

- Rays diverge from a single point (like light entering a camera)
- FOV is fixed in angular units (degrees)
- Higher altitude → wider ground coverage, more spread between hits
- Lower altitude → tighter ground coverage, denser hits
- Good for: simulating depth cameras, LiDAR with angular resolution

**Pattern Comparison**:

============== ==================== ==========================
Aspect         Grid                 Pinhole
============== ==================== ==========================
Ray direction  All parallel         Diverge from origin
Spacing        Meters               Degrees (FOV)
Height affect  No                   Yes
Real-world     Orthographic proj.   Perspective camera / LiDAR
============== ==================== ==========================

The pinhole behavior matches real depth sensors (RealSense, LiDAR) - when
you're farther from an object, each pixel covers more area.


Frame Attachment
----------------

Rays are attached to a frame in the scene via ``ObjRef``. Supported frame types:

- **body**: Attach to a body's origin. Rays follow body position and orientation.
- **site**: Attach to a site. Useful for precise placement or offset from body.
- **geom**: Attach to a geometry. Useful for sensors mounted on specific parts.

Example::

    from mjlab.sensor import ObjRef, RayCastSensorCfg, GridPatternCfg

    cfg = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(size=(1.0, 1.0), resolution=0.1),
    )

The ``exclude_parent_body`` option (default: True) prevents rays from hitting
the body they're attached to.


Ray Alignment
-------------

The ``ray_alignment`` setting controls how rays orient relative to the frame::

    Robot tilted 30°:

    "base" (default)          "yaw"                    "world"
    Rays tilt with body       Rays stay level          Rays fixed to world
          ↘ ↓ ↙                    ↓ ↓ ↓                    ↓ ↓ ↓
           \\|/                     |||                      |||
            🤖  ← tilted            🤖  ← tilted             🤖  ← tilted
           /                       /                        /

- **base**: Full position + rotation. Rays rotate with the body. Good for
  body-mounted sensors that should scan relative to the robot's orientation.

- **yaw**: Position + yaw only, ignores pitch/roll. Rays always point straight
  down regardless of body tilt. Good for height maps where you want consistent
  vertical sampling even when the robot is on a slope.

- **world**: Fixed in world frame, only position follows body. Rays always
  point in a fixed world direction. Good for gravity-aligned measurements.


Debug Visualization
-------------------

Enable visualization with ``debug_vis=True`` and customize via ``VizCfg``::

    cfg = RayCastSensorCfg(
        name="scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(),
        debug_vis=True,
        viz=RayCastSensorCfg.VizCfg(
            hit_color=(0, 1, 0, 0.8),      # Green for hits
            miss_color=(1, 0, 0, 0.4),     # Red for misses
            show_rays=True,                 # Draw ray arrows
            show_normals=True,              # Draw surface normals
            normal_color=(1, 1, 0, 1),     # Yellow normals
        ),
    )

Visualization options:

- ``hit_color`` / ``miss_color``: RGBA colors for ray arrows
- ``hit_sphere_color`` / ``hit_sphere_radius``: Spheres at hit points
- ``show_rays``: Draw arrows from origin to hit/miss points
- ``show_normals`` / ``normal_color`` / ``normal_length``: Surface normal arrows


Geom Group Filtering
--------------------

MuJoCo geoms can be assigned to groups 0-5. Use ``include_geom_groups`` to
filter which groups the rays can hit::

    cfg = RayCastSensorCfg(
        name="terrain_only",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(),
        include_geom_groups=(0, 1),  # Only hit geoms in groups 0 and 1
    )

This is useful for ignoring certain geometry (e.g., visual-only geoms in
group 3) while still detecting collisions with terrain (group 0).


Output Data
-----------

Access sensor data via the ``data`` property, which returns ``RayCastData``:

- ``distances``: [B, N] Distance to hit, or -1 if no hit / beyond max_distance
- ``hit_pos_w``: [B, N, 3] World-space hit positions
- ``normals_w``: [B, N, 3] Surface normals at hit points (world frame)
- ``pos_w``: [B, 3] Sensor frame position
- ``quat_w``: [B, 4] Sensor frame orientation (w, x, y, z)

Where B = number of environments, N = number of rays.
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
  from mjlab.viewer.debug_visualizer import DebugVisualizer


# NOTE: Need to define this here because it's not publicly exposed by mujoco_warp.
vec6 = wp.types.vector(length=6, dtype=float)

# Type aliases for configuration choices.
RayAlignment = Literal["base", "yaw", "world"]


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


PatternCfg = GridPatternCfg | PinholeCameraPatternCfg


@dataclass
class RayCastData:
  """Raycast sensor output data."""

  distances: torch.Tensor
  """[B, N] Distance to hit point. -1 if no hit."""

  normals_w: torch.Tensor
  """[B, N, 3] Surface normal at hit point (world frame). Zero if no hit."""

  hit_pos_w: torch.Tensor
  """[B, N, 3] Hit position in world frame. Zero if no hit."""

  pos_w: torch.Tensor
  """[B, 3] Frame position in world coordinates."""

  quat_w: torch.Tensor
  """[B, 4] Frame orientation quaternion (w, x, y, z) in world coordinates."""


@dataclass
class RayCastSensorCfg(SensorCfg):
  """Raycast sensor configuration.

  Supports multiple ray patterns (grid, pinhole camera) and alignment modes.
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

  frame: ObjRef
  """Body or site to attach rays to."""

  pattern: PatternCfg = field(default_factory=GridPatternCfg)
  """Ray pattern configuration. Defaults to GridPatternCfg."""

  ray_alignment: RayAlignment = "base"
  """How rays align with the frame.

  - "base": Full position + rotation (default).
  - "yaw": Position + yaw only, ignores pitch/roll (good for height maps).
  - "world": Fixed in world frame, position only follows body.
  """

  max_distance: float = 10.0
  """Maximum ray distance. Rays beyond this report -1."""

  exclude_parent_body: bool = True
  """Exclude parent body from ray intersection tests."""

  include_geom_groups: tuple[int, ...] | None = None
  """Geom groups (0-5) to include in raycasting. None means all groups."""

  debug_vis: bool = False
  """Enable debug visualization."""

  viz: VizCfg = field(default_factory=VizCfg)
  """Visualization settings."""

  def build(self) -> RayCastSensor:
    return RayCastSensor(self)


class RayCastSensor(Sensor[RayCastData]):
  """Raycast sensor for terrain and obstacle detection."""

  def __init__(self, cfg: RayCastSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self._data: mjwarp.Data | None = None
    self._model: mjwarp.Model | None = None
    self._mj_model: mujoco.MjModel | None = None
    self._device: str | None = None
    self._wp_device: wp.context.Device | None = None

    self._frame_body_id: int | None = None
    self._frame_site_id: int | None = None
    self._frame_geom_id: int | None = None
    self._frame_type: Literal["body", "site", "geom"] = "body"

    self._local_offsets: torch.Tensor | None = None
    self._local_directions: torch.Tensor | None = None  # [N, 3] per-ray directions
    self._num_rays: int = 0

    self._ray_pnt: wp.array | None = None
    self._ray_vec: wp.array | None = None
    self._ray_dist: wp.array | None = None
    self._ray_geomid: wp.array | None = None
    self._ray_normal: wp.array | None = None
    self._ray_bodyexclude: wp.array | None = None
    self._geomgroup = vec6(-1, -1, -1, -1, -1, -1)

    self._distances: torch.Tensor | None = None
    self._normals_w: torch.Tensor | None = None
    self._hit_pos_w: torch.Tensor | None = None
    self._pos_w: torch.Tensor | None = None
    self._quat_w: torch.Tensor | None = None

    self._raycast_graph: wp.Graph | None = None
    self._use_cuda_graph: bool = False

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

    frame = self.cfg.frame
    frame_name = frame.prefixed_name()

    if frame.type == "body":
      self._frame_body_id = mj_model.body(frame_name).id
      self._frame_type = "body"
    elif frame.type == "site":
      self._frame_site_id = mj_model.site(frame_name).id
      # Look up parent body for exclusion.
      self._frame_body_id = int(mj_model.site_bodyid[self._frame_site_id])
      self._frame_type = "site"
    elif frame.type == "geom":
      self._frame_geom_id = mj_model.geom(frame_name).id
      # Look up parent body for exclusion.
      self._frame_body_id = int(mj_model.geom_bodyid[self._frame_geom_id])
      self._frame_type = "geom"
    else:
      raise ValueError(
        f"RayCastSensor frame must be 'body', 'site', or 'geom', got '{frame.type}'"
      )

    # Generate ray pattern.
    pattern = self.cfg.pattern
    self._local_offsets, self._local_directions = pattern.generate_rays(
      mj_model, device
    )
    self._num_rays = self._local_offsets.shape[0]

    self._ray_pnt = wp.zeros((num_envs, self._num_rays), dtype=wp.vec3, device=device)
    self._ray_vec = wp.zeros((num_envs, self._num_rays), dtype=wp.vec3, device=device)
    self._ray_dist = wp.zeros((num_envs, self._num_rays), dtype=float, device=device)
    self._ray_geomid = wp.zeros((num_envs, self._num_rays), dtype=int, device=device)
    self._ray_normal = wp.zeros(
      (num_envs, self._num_rays), dtype=wp.vec3, device=device
    )

    body_exclude = (
      self._frame_body_id
      if self.cfg.exclude_parent_body and self._frame_body_id is not None
      else -1
    )
    self._ray_bodyexclude = wp.full(
      (self._num_rays,),
      body_exclude,
      dtype=int,  # pyright: ignore[reportArgumentType]
      device=device,
    )

    # Convert include_geom_groups to vec6 format (-1 = include, 0 = exclude).
    if self.cfg.include_geom_groups is not None:
      groups = [0, 0, 0, 0, 0, 0]
      for g in self.cfg.include_geom_groups:
        if 0 <= g <= 5:
          groups[g] = -1
      self._geomgroup = vec6(*groups)
    else:
      self._geomgroup = vec6(-1, -1, -1, -1, -1, -1)  # All groups

    assert self._wp_device is not None
    self._use_cuda_graph = self._wp_device.is_cuda and wp.is_mempool_enabled(
      self._wp_device
    )
    if self._use_cuda_graph:
      self._create_graph()

  def _compute_data(self) -> RayCastData:
    self._perform_raycast()
    assert self._distances is not None and self._normals_w is not None
    assert self._hit_pos_w is not None
    assert self._pos_w is not None and self._quat_w is not None
    return RayCastData(
      distances=self._distances,
      normals_w=self._normals_w,
      hit_pos_w=self._hit_pos_w,
      pos_w=self._pos_w,
      quat_w=self._quat_w,
    )

  @property
  def num_rays(self) -> int:
    return self._num_rays

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self.cfg.debug_vis:
      return
    assert self._data is not None
    assert self._local_offsets is not None
    assert self._local_directions is not None

    env_idx = visualizer.env_idx
    data = self.data

    if self._frame_type == "body":
      frame_pos = self._data.xpos[env_idx, self._frame_body_id].cpu().numpy()
      frame_mat_tensor = self._data.xmat[env_idx, self._frame_body_id].view(3, 3)
    elif self._frame_type == "site":
      frame_pos = self._data.site_xpos[env_idx, self._frame_site_id].cpu().numpy()
      frame_mat_tensor = self._data.site_xmat[env_idx, self._frame_site_id].view(3, 3)
    else:  # geom
      frame_pos = self._data.geom_xpos[env_idx, self._frame_geom_id].cpu().numpy()
      frame_mat_tensor = self._data.geom_xmat[env_idx, self._frame_geom_id].view(3, 3)

    # Apply ray alignment for visualization.
    rot_mat_tensor = self._compute_alignment_rotation(frame_mat_tensor.unsqueeze(0))[0]
    rot_mat = rot_mat_tensor.cpu().numpy()

    local_offsets_np = self._local_offsets.cpu().numpy()
    local_dirs_np = self._local_directions.cpu().numpy()
    hit_positions_np = data.hit_pos_w[env_idx].cpu().numpy()
    distances_np = data.distances[env_idx].cpu().numpy()
    normals_np = data.normals_w[env_idx].cpu().numpy()

    meansize = visualizer.meansize
    ray_width = 0.1 * meansize
    sphere_radius = self.cfg.viz.hit_sphere_radius * meansize
    normal_length = self.cfg.viz.normal_length * meansize
    normal_width = 0.1 * meansize

    for i in range(self._num_rays):
      origin = frame_pos + rot_mat @ local_offsets_np[i]
      hit = distances_np[i] >= 0

      if hit:
        end = hit_positions_np[i]
        color = self.cfg.viz.hit_color
      else:
        direction = rot_mat @ local_dirs_np[i]
        end = origin + direction * min(0.5, self.cfg.max_distance * 0.05)
        color = self.cfg.viz.miss_color

      if self.cfg.viz.show_rays:
        visualizer.add_arrow(
          start=origin,
          end=end,
          color=color,
          width=ray_width,
          label=f"{self.cfg.name}_ray_{i}",
        )

      if hit:
        visualizer.add_sphere(
          center=end,
          radius=sphere_radius,
          color=self.cfg.viz.hit_sphere_color,
          label=f"{self.cfg.name}_hit_{i}",
        )
        if self.cfg.viz.show_normals:
          normal_end = end + normals_np[i] * normal_length
          visualizer.add_arrow(
            start=end,
            end=normal_end,
            color=self.cfg.viz.normal_color,
            width=normal_width,
            label=f"{self.cfg.name}_normal_{i}",
          )

  # Private methods.

  def _create_graph(self) -> None:
    """Capture CUDA graph for raycast operation."""
    assert self._wp_device is not None and self._wp_device.is_cuda
    with wp.ScopedDevice(self._wp_device):
      with wp.ScopedCapture() as capture:
        self._raycast_direct()
      self._raycast_graph = capture.graph

  def _raycast_direct(self) -> None:
    """Execute raycast kernel directly."""
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
    )

  def _perform_raycast(self) -> None:
    assert self._data is not None and self._model is not None
    assert self._local_offsets is not None and self._local_directions is not None

    if self._frame_type == "body":
      frame_pos = self._data.xpos[:, self._frame_body_id]
      frame_mat = self._data.xmat[:, self._frame_body_id].view(-1, 3, 3)
    elif self._frame_type == "site":
      frame_pos = self._data.site_xpos[:, self._frame_site_id]
      frame_mat = self._data.site_xmat[:, self._frame_site_id].view(-1, 3, 3)
    else:  # geom
      frame_pos = self._data.geom_xpos[:, self._frame_geom_id]
      frame_mat = self._data.geom_xmat[:, self._frame_geom_id].view(-1, 3, 3)

    num_envs = frame_pos.shape[0]

    # Apply ray alignment.
    rot_mat = self._compute_alignment_rotation(frame_mat)

    # Transform ray origins.
    world_offsets = torch.einsum("bij,nj->bni", rot_mat, self._local_offsets)
    world_origins = frame_pos.unsqueeze(1) + world_offsets

    # Transform ray directions (per-ray).
    world_rays = torch.einsum("bij,nj->bni", rot_mat, self._local_directions)

    assert self._ray_pnt is not None and self._ray_vec is not None
    pnt_torch = wp.to_torch(self._ray_pnt).view(num_envs, self._num_rays, 3)
    vec_torch = wp.to_torch(self._ray_vec).view(num_envs, self._num_rays, 3)
    pnt_torch.copy_(world_origins)
    vec_torch.copy_(world_rays)

    if self._use_cuda_graph and self._raycast_graph is not None:
      with wp.ScopedDevice(self._wp_device):
        wp.capture_launch(self._raycast_graph)
    else:
      self._raycast_direct()

    assert self._ray_dist is not None and self._ray_normal is not None
    self._distances = wp.to_torch(self._ray_dist)
    self._normals_w = wp.to_torch(self._ray_normal).view(num_envs, self._num_rays, 3)
    self._distances[self._distances > self.cfg.max_distance] = -1.0

    # Compute hit positions: origin + direction * distance.
    # For misses (distance = -1), hit_pos_w will be invalid (but normals_w are zero).
    assert self._distances is not None
    hit_mask = self._distances >= 0
    hit_pos_w = world_origins.clone()
    hit_pos_w[hit_mask] = world_origins[hit_mask] + world_rays[
      hit_mask
    ] * self._distances[hit_mask].unsqueeze(-1)
    self._hit_pos_w = hit_pos_w

    self._pos_w = frame_pos.clone()
    self._quat_w = quat_from_matrix(frame_mat)

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
