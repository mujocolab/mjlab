from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor


@wp.kernel
def _unpack_rgb_kernel(
  packed: wp.array2d(dtype=wp.uint32),  # type: ignore
  rgb: wp.array3d(dtype=wp.uint8),  # type: ignore
):
  world_idx, pixel_idx = wp.tid()  # type: ignore
  b = wp.uint8(packed[world_idx, pixel_idx] & wp.uint32(0xFF))
  g = wp.uint8((packed[world_idx, pixel_idx] >> wp.uint32(8)) & wp.uint32(0xFF))
  r = wp.uint8((packed[world_idx, pixel_idx] >> wp.uint32(16)) & wp.uint32(0xFF))
  rgb[world_idx, pixel_idx, 0] = r
  rgb[world_idx, pixel_idx, 1] = g
  rgb[world_idx, pixel_idx, 2] = b


class RenderManager:
  """Manages rendering for all camera sensors in a scene.

  Coordinates rendering across multiple camera sensors using a 2-level control system:

  1. **Static enablement (cam_active)**: Only cameras with corresponding sensors
    are included in the render context at initialization. This determines the set
    of cameras that CAN be rendered. This is useful for scenarios where an XML file
    may contain many cameras, but only a subset are used as sensors. In this case,
    the other cameras are disabled.

  2. **Dynamic toggle (render_rgb/render_depth)**: On each render() call, these
    flags control which enabled cameras ACTUALLY render based on whether their
    sensor cache is invalid, allowing selective rendering.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    camera_sensors: list[CameraSensor],
    device: str,
  ):
    self.wp_device = wp.get_device(device)

    # Sort sensors by camera index to match mujoco_warp ordering.
    self.camera_sensors = sorted(camera_sensors, key=lambda s: s.camera_idx)
    camera_resolutions = [(s.cfg.width, s.cfg.height) for s in self.camera_sensors]
    render_rgb = ["rgb" in s.cfg.type for s in self.camera_sensors]
    render_depth = ["depth" in s.cfg.type for s in self.camera_sensors]

    # Create mapping from MuJoCo camera ID to sorted list index.
    self._cam_idx_to_list_idx = {
      s.camera_idx: idx for idx, s in enumerate(self.camera_sensors)
    }

    # Validate that all sensors have consistent rendering settings.
    first_sensor = self.camera_sensors[0]
    use_textures = first_sensor.cfg.use_textures
    use_shadows = first_sensor.cfg.use_shadows
    enabled_geom_groups = list(first_sensor.cfg.enabled_geom_groups)

    for sensor in self.camera_sensors[1:]:
      if sensor.cfg.use_textures != use_textures:
        raise ValueError(
          f"Camera sensor '{sensor.cfg.name}' has "
          f"use_textures={sensor.cfg.use_textures}, "
          f"but '{first_sensor.cfg.name}' has use_textures={use_textures}. "
          "All camera sensors must have the same use_textures setting."
        )
      if sensor.cfg.use_shadows != use_shadows:
        raise ValueError(
          f"Camera sensor '{sensor.cfg.name}' has "
          f"use_shadows={sensor.cfg.use_shadows}, "
          f"but '{first_sensor.cfg.name}' has use_shadows={use_shadows}. "
          "All camera sensors must have the same use_shadows setting."
        )
      if tuple(sensor.cfg.enabled_geom_groups) != tuple(enabled_geom_groups):
        raise ValueError(
          f"Camera sensor '{sensor.cfg.name}' has enabled_geom_groups="
          f"{sensor.cfg.enabled_geom_groups}, but '{first_sensor.cfg.name}' has "
          f"enabled_geom_groups={tuple(enabled_geom_groups)}. "
          "All camera sensors must have the same enabled_geom_groups setting."
        )

    # Build cam_active list: mark only cameras with sensors as active.
    cam_active = [False] * mj_model.ncam
    for sensor in self.camera_sensors:
      cam_active[sensor.camera_idx] = True

    with wp.ScopedDevice(self.wp_device):
      self._ctx = mjwarp.create_render_context(
        mjm=mj_model,
        m=model.struct,  # type: ignore[attr-defined]
        d=data.struct,  # type: ignore[attr-defined]
        cam_res=camera_resolutions,
        render_rgb=render_rgb,
        render_depth=render_depth,
        use_textures=use_textures,
        use_shadows=use_shadows,
        enabled_geom_groups=enabled_geom_groups,
        cam_active=cam_active,
      )

    self._model = model
    self._data = data
    self._camera_resolutions = camera_resolutions
    self._render_rgb = render_rgb
    self._render_depth = render_depth
    self._use_shadows = use_shadows
    self._use_textures = use_textures
    self._enabled_geom_groups = enabled_geom_groups
    self._cam_active = cam_active
    self._rgb_adr = self._ctx.rgb_adr.numpy()
    self._depth_adr = self._ctx.depth_adr.numpy()
    self._rgb_size = self._ctx.rgb_size.numpy()
    self._depth_size = self._ctx.depth_size.numpy()
    self._render_rgb_torch = wp.to_torch(self._ctx.render_rgb)
    self._render_depth_torch = wp.to_torch(self._ctx.render_depth)

    if any(render_rgb):
      self._rgb_unpacked = wp.array3d(
        shape=(data.nworld, self._ctx.rgb_data.shape[1], 3),
        dtype=wp.uint8,
        device=self.wp_device,
      )
    else:
      self._rgb_unpacked = None

    for sensor in self.camera_sensors:
      sensor.set_render_manager(self)

    self.use_cuda_graph = self.wp_device.is_cuda and wp.is_mempool_enabled(
      self.wp_device
    )
    self.create_graph()
    self._needs_render = True

  def create_graph(self) -> None:
    self.render_graph = None
    if self.use_cuda_graph:
      with wp.ScopedDevice(self.wp_device):
        with wp.ScopedCapture() as capture:
          mjwarp.render(self._model, self._data, self._ctx)
        self.render_graph = capture.graph
  
  def recreate_render_context(self, mj_model: mujoco.MjModel) -> None:
    with wp.ScopedDevice(self.wp_device):
      self._ctx = mjwarp.create_render_context(
        mjm=mj_model,
        m=self._model.struct,  # type: ignore
        d=self._data.struct,  # type: ignore
        cam_res=self._camera_resolutions,
        render_rgb=self._render_rgb,
        render_depth=self._render_depth,
        use_textures=self._use_textures,
        use_shadows=self._use_shadows,
        enabled_geom_groups=self._enabled_geom_groups,
        cam_active=self._cam_active,
      )

  def invalidate(self) -> None:
    """Mark that rendering is needed on next data access."""
    self._needs_render = True

  def ensure_rendered(self) -> None:
    """Render if invalidated. Called automatically on data access."""
    if not self._needs_render:
      return
    self._do_render()
    self._needs_render = False

  def _do_render(self) -> None:
    any_render_needed = False
    any_rgb_rendered = False

    for idx, sensor in enumerate(self.camera_sensors):
      should_render = not sensor._cache_valid
      should_render_rgb = should_render and ("rgb" in sensor.cfg.type)
      should_render_depth = should_render and ("depth" in sensor.cfg.type)

      self._render_rgb_torch[idx] = should_render_rgb
      self._render_depth_torch[idx] = should_render_depth

      if should_render_rgb or should_render_depth:
        any_render_needed = True
      if should_render_rgb:
        any_rgb_rendered = True

    if not any_render_needed:
      return

    with wp.ScopedDevice(self.wp_device):
      if self.use_cuda_graph and self.render_graph is not None:
        wp.capture_launch(self.render_graph)
      else:
        mjwarp.render(self._model, self._data, self._ctx)

    if any_rgb_rendered and self._rgb_unpacked is not None:
      wp.launch(
        _unpack_rgb_kernel,
        dim=(self._data.nworld, self._ctx.rgb_data.shape[1]),
        inputs=[self._ctx.rgb_data],
        outputs=[self._rgb_unpacked],
        device=self.wp_device,
      )

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    if self._rgb_unpacked is None:
      raise RuntimeError(
        "RGB rendering is not enabled. Ensure at least one camera sensor has "
        "'rgb' in its type configuration."
      )

    if cam_idx not in self._cam_idx_to_list_idx:
      available = list(self._cam_idx_to_list_idx.keys())
      raise KeyError(
        f"Camera ID {cam_idx} not found in RenderManager. "
        f"Available camera IDs: {available}"
      )

    # Validate this specific camera has RGB enabled.
    list_idx = self._cam_idx_to_list_idx[cam_idx]
    if not self._render_rgb[list_idx]:
      raise RuntimeError(
        f"Camera ID {cam_idx} does not have RGB rendering enabled. "
        f"Ensure 'rgb' is in the sensor's type configuration."
      )

    rgb_unpacked_torch = wp.to_torch(self._rgb_unpacked)
    start = int(self._rgb_adr[list_idx])
    size = int(self._rgb_size[list_idx])
    rgb_flat = rgb_unpacked_torch[:, start : start + size]
    return rgb_flat.reshape(
      self._data.nworld,
      self.camera_sensors[list_idx].cfg.height,
      self.camera_sensors[list_idx].cfg.width,
      3,
    )

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    if not any(self._render_depth):
      raise RuntimeError(
        "Depth rendering is not enabled. Ensure at least one camera sensor has "
        "'depth' in its type configuration."
      )

    if cam_idx not in self._cam_idx_to_list_idx:
      available = list(self._cam_idx_to_list_idx.keys())
      raise KeyError(
        f"Camera ID {cam_idx} not found in RenderManager. "
        f"Available camera IDs: {available}"
      )

    # Validate this specific camera has depth enabled.
    list_idx = self._cam_idx_to_list_idx[cam_idx]
    if not self._render_depth[list_idx]:
      raise RuntimeError(
        f"Camera ID {cam_idx} does not have depth rendering enabled. "
        f"Ensure 'depth' is in the sensor's type configuration."
      )

    depth_torch = wp.to_torch(self._ctx.depth_data)
    start = int(self._depth_adr[list_idx])
    size = int(self._depth_size[list_idx])
    depth_flat = depth_torch[:, start : start + size]
    return depth_flat.reshape(
      self._data.nworld,
      self.camera_sensors[list_idx].cfg.height,
      self.camera_sensors[list_idx].cfg.width,
      1,
    )
