from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
  from mjlab.sensor.render_manager import RenderManager

CameraDataType = Literal["rgb", "depth"]

# Default MuJoCo fov, in degrees.
_DEFAULT_CAM_FOVY = 45.0


@dataclass
class CameraSensorCfg(SensorCfg):
  """Configuration for a camera sensor.

  Note:
      All camera sensors in a scene must share identical values for
      use_textures, use_shadows, and enabled_geom_groups. This is a
      constraint of the underlying mujoco_warp rendering system.
  """

  camera_name: str | None = None
  pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  fovy: float | None = None
  width: int = 160
  height: int = 120
  type: tuple[CameraDataType, ...] = ("rgb",)
  use_textures: bool = True
  use_shadows: bool = False
  enabled_geom_groups: tuple[int, ...] = (0, 1, 2)
  clone_data: bool = False
  """If True, clone tensors on each access. Set to True if you modify the data."""

  def build(self) -> CameraSensor:
    return CameraSensor(self)


@dataclass
class CameraSensorData:
  """Camera sensor output data.

  Contains RGB and/or depth images based on sensor configuration.
  Both fields are None if not requested in the sensor's type configuration.

  Shape: [num_envs, height, width, channels]
    - rgb: channels=3 (uint8)
    - depth: channels=1 (float32)
  """

  rgb: torch.Tensor | None = None
  """RGB image data [num_envs, height, width, 3] (uint8). None if not enabled."""
  depth: torch.Tensor | None = None
  """Depth image data [num_envs, height, width, 1] (float32). None if not enabled."""


class CameraSensor(Sensor[CameraSensorData]):
  def __init__(self, cfg: CameraSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self._camera_name = cfg.camera_name if cfg.camera_name is not None else cfg.name
    self._is_wrapping_existing = cfg.camera_name is not None
    self._render_manager: RenderManager | None = None
    self._camera_idx: int = -1

  @property
  def camera_name(self) -> str:
    return self._camera_name

  @property
  def camera_idx(self) -> int:
    return self._camera_idx

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del entities

    if self._is_wrapping_existing:
      if self.cfg.fovy is not None:
        scene_spec.camera(self._camera_name).fovy = self.cfg.fovy
      return

    scene_spec.worldbody.add_camera(
      name=self.cfg.name,
      pos=self.cfg.pos,
      quat=self.cfg.quat,
      fovy=self.cfg.fovy or _DEFAULT_CAM_FOVY,
    )

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    del model, data, device

    try:
      cam = mj_model.camera(self._camera_name)
      self._camera_idx = cam.id
    except KeyError as e:
      available = [mj_model.cam(i).name for i in range(mj_model.ncam)]
      raise ValueError(
        f"Camera '{self._camera_name}' not found in model. Available: {available}"
      ) from e

  def set_render_manager(self, render_manager: RenderManager) -> None:
    self._render_manager = render_manager

  def _compute_data(self) -> CameraSensorData:
    assert self._render_manager is not None
    self._render_manager.ensure_rendered()
    rgb_data = None
    depth_data = None
    if "rgb" in self.cfg.type:
      rgb = self._render_manager.get_rgb(self._camera_idx)
      rgb_data = rgb.clone() if self.cfg.clone_data else rgb
    if "depth" in self.cfg.type:
      depth = self._render_manager.get_depth(self._camera_idx)
      depth_data = depth.clone() if self.cfg.clone_data else depth
    return CameraSensorData(rgb=rgb_data, depth=depth_data)
