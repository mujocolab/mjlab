"""Shared render context for camera and raycast sensors."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor
  from mjlab.sensor.raycast_sensor import RayCastSensor


@wp.kernel
def _unpack_rgb_kernel(
  packed: wp.array2d(dtype=wp.uint32),  # type: ignore[valid-type]
  rgb: wp.array3d(dtype=wp.uint8),  # type: ignore[valid-type]
):
  """Unpack ABGR uint32 pixels into separate R, G, B uint8 channels."""
  world_idx, pixel_idx = wp.tid()  # type: ignore[attr-defined]
  val = packed[world_idx, pixel_idx]
  b = wp.uint8(val & wp.uint32(0xFF))
  g = wp.uint8((val >> wp.uint32(8)) & wp.uint32(0xFF))
  r = wp.uint8((val >> wp.uint32(16)) & wp.uint32(0xFF))
  rgb[world_idx, pixel_idx, 0] = r
  rgb[world_idx, pixel_idx, 1] = g
  rgb[world_idx, pixel_idx, 2] = b


# Fields that require per-pixel ray recomputation at render time.
_RAY_FIELDS = frozenset({"cam_fovy", "cam_intrinsic"})


class SensorContext:
  """Container for shared sensing resources.

  Manages the RenderContext used by both camera sensors (for rendering) and raycast
  sensors (for BVH-accelerated ray intersection). The actual graph capture and
  execution is handled by Simulation.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    camera_sensors: list[CameraSensor],
    raycast_sensors: list[RayCastSensor],
    device: str,
  ):
    self._model = model
    self._data = data
    self.wp_device = wp.get_device(device)

    # Sort camera sensors by camera index for consistent ordering.
    self.camera_sensors = sorted(camera_sensors, key=lambda s: s.camera_idx)
    self.raycast_sensors = raycast_sensors

    # MuJoCo camera ID -> sorted list index.
    self._cam_idx_to_list_idx = {
      s.camera_idx: idx for idx, s in enumerate(self.camera_sensors)
    }

    if self.camera_sensors:
      self._validate_sensor_settings()

    self._rgb_unpacked: wp.array | None = None
    self._rgb_torch: torch.Tensor | None = None
    self._depth_torch: torch.Tensor | None = None
    self._rgb_adr_np: list[int] | None = None
    self._depth_adr_np: list[int] | None = None
    self._disable_precomputed_rays = False
    self._create_context(mj_model)

    # Wire up sensors to use this context.
    for sensor in self.camera_sensors:
      sensor.set_context(self)
    for sensor in self.raycast_sensors:
      sensor.set_context(self)

  @property
  def has_cameras(self) -> bool:
    return len(self.camera_sensors) > 0

  @property
  def has_raycasts(self) -> bool:
    return len(self.raycast_sensors) > 0

  @property
  def render_context(self) -> mjwarp.RenderContext:
    return self._ctx

  def recreate(
    self,
    mj_model: mujoco.MjModel,
    expanded_fields: set[str] | None = None,
  ) -> None:
    """Recreate the render context after model fields are expanded.

    Called by Simulation.expand_model_fields() for domain randomization.
    """
    if expanded_fields is not None:
      self._disable_precomputed_rays = bool(expanded_fields & _RAY_FIELDS)
    self._create_context(mj_model)

  def prepare(self) -> None:
    """Pre-graph: transform rays to world frame."""
    for sensor in self.raycast_sensors:
      sensor.prepare_rays()

  def finalize(self) -> None:
    """Post-graph: compute raycast hit positions."""
    for sensor in self.raycast_sensors:
      sensor.postprocess_rays()

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    """Get unpacked RGB data for a camera.

    Args:
      cam_idx: MuJoCo camera ID.

    Returns:
      Tensor of shape [num_envs, height, width, 3] (uint8).
    """
    if self._rgb_unpacked is None:
      raise RuntimeError(
        "RGB rendering is not enabled. Ensure at least one camera sensor has 'rgb' in"
        " its data_types."
      )

    if cam_idx not in self._cam_idx_to_list_idx:
      available = list(self._cam_idx_to_list_idx.keys())
      raise KeyError(
        f"Camera ID {cam_idx} not found in SensorContext. "
        f"Available camera IDs: {available}"
      )

    list_idx = self._cam_idx_to_list_idx[cam_idx]

    assert self._rgb_adr_np is not None
    assert self._rgb_torch is not None
    rgb_adr = self._rgb_adr_np[list_idx]
    if rgb_adr < 0:
      raise RuntimeError(f"Camera ID {cam_idx} does not have RGB rendering enabled.")

    sensor = self.camera_sensors[list_idx]
    w, h = sensor.cfg.width, sensor.cfg.height
    num_pixels = w * h
    nworld = self._data.nworld

    cam_data = self._rgb_torch[:, rgb_adr : rgb_adr + num_pixels, :]
    return cam_data.view(nworld, h, w, 3)

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    """Get depth data for a camera.

    Args:
      cam_idx: MuJoCo camera ID.

    Returns:
      Tensor of shape [num_envs, height, width, 1] (float32).
    """
    if cam_idx not in self._cam_idx_to_list_idx:
      available = list(self._cam_idx_to_list_idx.keys())
      raise KeyError(
        f"Camera ID {cam_idx} not found in SensorContext. "
        f"Available camera IDs: {available}"
      )

    list_idx = self._cam_idx_to_list_idx[cam_idx]

    assert self._depth_adr_np is not None
    assert self._depth_torch is not None
    depth_adr = self._depth_adr_np[list_idx]
    if depth_adr < 0:
      raise RuntimeError(f"Camera ID {cam_idx} does not have depth rendering enabled.")

    sensor = self.camera_sensors[list_idx]
    w, h = sensor.cfg.width, sensor.cfg.height
    num_pixels = w * h
    nworld = self._data.nworld

    cam_data = self._depth_torch[:, depth_adr : depth_adr + num_pixels]
    return cam_data.view(nworld, h, w, 1)

  # Private methods.

  def _validate_sensor_settings(self) -> None:
    """Validate that all camera sensors share render settings."""
    ref = self.camera_sensors[0].cfg
    for s in self.camera_sensors[1:]:
      cfg = s.cfg
      if cfg.use_textures != ref.use_textures:
        raise ValueError(
          "All camera sensors must share the same use_textures "
          f"setting. '{s.cfg.name}' differs from "
          f"'{ref.name}'."
        )
      if cfg.use_shadows != ref.use_shadows:
        raise ValueError(
          "All camera sensors must share the same use_shadows "
          f"setting. '{s.cfg.name}' differs from "
          f"'{ref.name}'."
        )
      if cfg.enabled_geom_groups != ref.enabled_geom_groups:
        raise ValueError(
          "All camera sensors must share the same "
          f"enabled_geom_groups. '{s.cfg.name}' differs from "
          f"'{ref.name}'."
        )

  def _raycast_geom_groups(self) -> set[int]:
    """Compute the union of geom groups across all raycast sensors."""
    groups: set[int] = set()
    for s in self.raycast_sensors:
      if s.cfg.include_geom_groups is None:
        return {0, 1, 2, 3, 4, 5}
      groups.update(s.cfg.include_geom_groups)
    return groups

  def _create_context(self, mj_model: mujoco.MjModel) -> None:
    """Create the mujoco_warp RenderContext."""
    ncam = mj_model.ncam
    cam_res: list[tuple[int, int]] | None = None
    render_rgb: list[bool] | None = None
    render_depth: list[bool] | None = None

    raycast_geom_groups = self._raycast_geom_groups()

    if self.camera_sensors:
      ref_cfg = self.camera_sensors[0].cfg
      use_textures = ref_cfg.use_textures
      use_shadows = ref_cfg.use_shadows
      cam_geom_groups = set(ref_cfg.enabled_geom_groups)

      if self.raycast_sensors and raycast_geom_groups != cam_geom_groups:
        merged = sorted(cam_geom_groups | raycast_geom_groups)
        warnings.warn(
          "Camera enabled_geom_groups "
          f"{sorted(cam_geom_groups)} and raycast "
          f"include_geom_groups {sorted(raycast_geom_groups)}"
          f" differ. Using union {merged}.",
          stacklevel=2,
        )
        enabled_geom_groups = merged
      else:
        enabled_geom_groups = sorted(cam_geom_groups)

      # Build cam_active mask: only activate cameras that are sensors.
      # cam_res, render_rgb, render_depth follow sorted sensor order (by camera_idx).
      # This must match the order used by create_render_context for rgb_adr/depth_adr
      # indexing.
      cam_active = [False] * ncam
      cam_res = []
      render_rgb = []
      render_depth = []

      for s in self.camera_sensors:
        cam_active[s.camera_idx] = True
        cam_res.append((s.cfg.width, s.cfg.height))
        render_rgb.append("rgb" in s.cfg.data_types)
        render_depth.append("depth" in s.cfg.data_types)
    else:
      # Raycasts-only: need BVH but no camera rendering.
      use_textures = False
      use_shadows = False
      enabled_geom_groups = sorted(raycast_geom_groups)
      cam_active = [False] * ncam

    with wp.ScopedDevice(self.wp_device):
      self._ctx = mjwarp.create_render_context(
        mjm=mj_model,
        nworld=self._data.nworld,
        cam_res=cam_res,
        render_rgb=render_rgb,
        render_depth=render_depth,
        use_textures=use_textures,
        use_shadows=use_shadows,
        enabled_geom_groups=enabled_geom_groups,
        cam_active=cam_active,
        use_precomputed_rays=not self._disable_precomputed_rays,
      )

    # Cache address arrays from the render context. An adr value of -1 means that data
    # type is not enabled for that camera.
    if self.camera_sensors:
      self._rgb_adr_np = self._ctx.rgb_adr.numpy().tolist()
      self._depth_adr_np = self._ctx.depth_adr.numpy().tolist()

    has_any_rgb = self._rgb_adr_np is not None and any(a >= 0 for a in self._rgb_adr_np)
    has_any_depth = self._depth_adr_np is not None and any(
      a >= 0 for a in self._depth_adr_np
    )

    # Allocate RGB unpack buffer if any camera wants RGB.
    if has_any_rgb:
      total_rgb_pixels = self._ctx.rgb_data.shape[1]
      nworld = self._data.nworld
      self._rgb_unpacked = wp.zeros(
        (nworld, total_rgb_pixels, 3),
        dtype=wp.uint8,
        device=self.wp_device,
      )
      self._rgb_torch = wp.to_torch(self._rgb_unpacked)
    else:
      self._rgb_unpacked = None
      self._rgb_torch = None

    # Cache depth torch view (zero-copy).
    if has_any_depth:
      self._depth_torch = wp.to_torch(self._ctx.depth_data)
    else:
      self._depth_torch = None

  def unpack_rgb(self) -> None:
    """Unpack packed uint32 RGB data into separate channels.

    Called from Simulation._sense_kernel() so it gets captured in the
    CUDA graph. No-op if no cameras need RGB.
    """
    if self._rgb_unpacked is not None:
      wp.launch(
        _unpack_rgb_kernel,
        dim=(self._data.nworld, self._ctx.rgb_data.shape[1]),
        inputs=[self._ctx.rgb_data],
        outputs=[self._rgb_unpacked],
        device=self.wp_device,
      )
