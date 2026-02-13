"""Camera image visualization for Viser viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import viser
import viser.transforms as vtf

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor


class ViserCameraViewer:
  """Handles camera image visualization for the Viser viewer."""

  def __init__(
    self,
    server: viser.ViserServer,
    camera_sensor: CameraSensor,
    mj_model: mujoco.MjModel,
    min_display_size: int = 128,
  ):
    self._server = server
    self._camera_sensor = camera_sensor
    self._mj_model = mj_model

    self._rgb_handle: viser.GuiImageHandle | None = None
    self._depth_handle: viser.GuiImageHandle | None = None
    self._frustum_handle: viser.CameraFrustumHandle | None = None

    self._camera_name = camera_sensor.camera_name
    self._camera_idx = camera_sensor.camera_idx

    self._has_rgb = "rgb" in self._camera_sensor.cfg.data_types
    self._has_depth = "depth" in self._camera_sensor.cfg.data_types

    height = self._camera_sensor.cfg.height
    width = self._camera_sensor.cfg.width

    scale = max(1, min_display_size // max(height, width))
    self._display_height = height * scale
    self._display_width = width * scale
    self._needs_upsampling = scale > 1

    if self._has_rgb:
      self._rgb_handle = self._server.gui.add_image(
        image=np.zeros((self._display_height, self._display_width, 3), dtype=np.uint8),
        label=f"{self._camera_name}_rgb",
        format="jpeg",
      )

    if self._has_depth:
      self._depth_scale_slider = self._server.gui.add_slider(
        label="Depth Scale",
        min=0.1,
        max=10.0,
        step=0.1,
        initial_value=1.0,
      )
      self._depth_handle = self._server.gui.add_image(
        image=np.zeros((self._display_height, self._display_width, 3), dtype=np.uint8),
        label=f"{self._camera_name}_depth",
        format="jpeg",
      )

    self._show_frustum_toggle = self._server.gui.add_checkbox(
      label="Frustum",
      initial_value=True,
    )
    self._fov, self._aspect = self._compute_camera_fov_aspect()

  def _compute_camera_fov_aspect(self) -> tuple[float, float]:
    cam_id = self._camera_idx
    fovy = self._mj_model.cam_fovy[cam_id]  # in degrees
    fovy_rad = np.deg2rad(fovy)
    aspect = self._camera_sensor.cfg.width / self._camera_sensor.cfg.height
    return fovy_rad, aspect

  def _update_frustum(self, sim_data, env_idx: int) -> None:
    if not self._show_frustum_toggle.value:
      if self._frustum_handle is not None:
        self._frustum_handle.remove()
        self._frustum_handle = None
      return

    # Get camera pose from simulation data
    cam_id = self._camera_idx
    cam_pos = sim_data.cam_xpos[env_idx, cam_id].cpu().numpy()  # [3]
    cam_mat = sim_data.cam_xmat[env_idx, cam_id].cpu().numpy().reshape(3, 3)  # [3, 3]

    # Convert rotation matrix to quaternion (wxyz format for viser)
    # MuJoCo camera looks along -Z axis, viser frustum expects looking along +Z
    # So we need to rotate 180 degrees around X axis
    rot_180_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    cam_mat_adjusted = cam_mat @ rot_180_x

    wxyz = vtf.SO3.from_matrix(cam_mat_adjusted).wxyz

    if self._frustum_handle is None:
      self._frustum_handle = self._server.scene.add_camera_frustum(
        name=f"/{self._camera_name}_frustum",
        fov=self._fov,
        aspect=self._aspect,
        position=cam_pos,
        wxyz=wxyz,
        scale=0.15,
        color=(200, 200, 200),
      )
    else:
      self._frustum_handle.position = cam_pos
      self._frustum_handle.wxyz = wxyz

  def _upsample_nearest(self, image: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

  def update(self, sim_data, env_idx: int = 0) -> None:
    data = self._camera_sensor.data

    if self._has_rgb and self._rgb_handle is not None and data.rgb is not None:
      rgb_np = data.rgb[env_idx].cpu().numpy()

      # Upsample if needed for better visibility
      if self._needs_upsampling:
        scale = self._display_height // rgb_np.shape[0]
        rgb_np = self._upsample_nearest(rgb_np, scale)

      self._rgb_handle.image = rgb_np

    if self._has_depth and self._depth_handle is not None and data.depth is not None:
      depth_np = data.depth[env_idx].squeeze().cpu().numpy()

      depth_scale = max(self._depth_scale_slider.value, 0.01)
      depth_normalized = np.clip(depth_np / depth_scale, 0.0, 1.0)
      depth_uint8 = (depth_normalized * 255).astype(np.uint8)
      if self._needs_upsampling:
        scale = self._display_height // depth_uint8.shape[0]
        depth_uint8 = self._upsample_nearest(depth_uint8, scale)

      self._depth_handle.image = np.repeat(depth_uint8[:, :, np.newaxis], 3, axis=-1)

    self._update_frustum(sim_data, env_idx)

  def cleanup(self) -> None:
    if self._rgb_handle is not None:
      self._rgb_handle.remove()
    if self._depth_handle is not None:
      self._depth_handle.remove()
      self._depth_scale_slider.remove()
    if self._frustum_handle is not None:
      self._frustum_handle.remove()
    self._show_frustum_toggle.remove()
