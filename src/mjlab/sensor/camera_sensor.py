"""Camera sensor for RGB and depth rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
  from mjlab.sensor.sensor_context import SensorContext

CameraDataType = Literal["rgb", "depth", "segmentation"]

# Default MuJoCo fov, in degrees.
_DEFAULT_CAM_FOVY = 45.0


@dataclass
class CameraSensorCfg(SensorCfg):
  """Configuration for a camera sensor.

  A camera sensor can either wrap an existing MuJoCo camera
  (``camera_name``) or create a new one at the specified
  pos/quat. New cameras are added to the worldbody by default,
  or to a specific body via ``parent_body``.

  Note:
    All camera sensors in a scene must share identical values for
    use_textures, use_shadows, and enabled_geom_groups. This is a
    constraint of the underlying mujoco_warp rendering system.
  """

  camera_name: str | None = None
  """Name of an existing MuJoCo camera to wrap.

  If None, a new camera is created using pos/quat/fovy.
  If set, the sensor wraps the named camera instead of creating one.
  """

  parent_body: str | None = None
  """Parent body to attach a new camera to.

  Only used when ``camera_name`` is None (creating a new camera).
  If None, the camera is added to the worldbody. Use the full
  prefixed name (e.g., "robot/link_6") to attach to an entity's
  body. The pos/quat are then relative to the parent body frame.
  """

  pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
  """Camera position (used when creating a new camera).

  World-frame if parent_body is None, otherwise relative to the
  parent body frame.
  """

  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  """Camera orientation quaternion (w, x, y, z)."""

  fovy: float | None = None
  """Vertical field of view in degrees. None uses MuJoCo default."""

  width: int = 160
  """Image width in pixels."""

  height: int = 120
  """Image height in pixels."""

  data_types: tuple[CameraDataType, ...] = ("rgb",)
  """Data types to capture: any combination of "rgb", "depth",
  and "segmentation"."""

  use_textures: bool = True
  """Whether to use textures in rendering."""

  use_shadows: bool = False
  """Whether to use shadows in rendering."""

  enabled_geom_groups: tuple[int, ...] = (0, 1, 2)
  """Geom groups (0-5) visible to the camera."""

  orthographic: bool = False
  """Use orthographic projection instead of perspective."""

  clone_data: bool = False
  """If True, clone tensors on each access.

  Set to True if you modify the returned data in-place.
  """

  def __post_init__(self) -> None:
    valid = {"rgb", "depth", "segmentation"}
    invalid = {dt for dt in self.data_types if dt not in valid}
    if invalid:
      raise ValueError(f"Invalid camera data types: {invalid}. Valid types: {valid}")
    if not self.data_types:
      raise ValueError("At least one data type must be specified.")

  def build(self) -> CameraSensor:
    return CameraSensor(self)


@dataclass
class CameraSensorData:
  """Camera sensor output data.

  Shapes:
    - rgb: [num_envs, height, width, 3] (uint8)
    - depth: [num_envs, height, width, 1] (float32)
    - segmentation: [num_envs, height, width, 1] (int32)
  """

  rgb: torch.Tensor | None = None
  """RGB image [num_envs, height, width, 3] (uint8). None if not
  enabled."""

  depth: torch.Tensor | None = None
  """Depth image [num_envs, height, width, 1] (float32). None if not
  enabled."""

  segmentation: torch.Tensor | None = None
  """Per-pixel geom IDs [num_envs, height, width, 1] (int32).

  Values: >= 0 for rigid geom IDs, -1 for background, -2 for flex
  bodies. None if not enabled.
  """


class CameraSensor(Sensor[CameraSensorData]):
  """Camera sensor for RGB and depth rendering."""

  requires_sensor_context = True

  def __init__(self, cfg: CameraSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self._camera_name = cfg.camera_name if cfg.camera_name is not None else cfg.name
    self._is_wrapping_existing = cfg.camera_name is not None
    self._ctx: SensorContext | None = None
    self._camera_idx: int = -1

  @property
  def camera_name(self) -> str:
    return self._camera_name

  @property
  def camera_idx(self) -> int:
    return self._camera_idx

  def edit_spec(
    self,
    scene_spec: mujoco.MjSpec,
    entities: dict[str, Entity],
  ) -> None:
    del entities

    if self._is_wrapping_existing:
      cam = scene_spec.camera(self._camera_name)
      assert isinstance(cam, mujoco.MjsCamera)
      if self.cfg.fovy is not None:
        cam.fovy = self.cfg.fovy
      if self.cfg.orthographic:
        cam.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
      return

    if self.cfg.parent_body is not None:
      parent = scene_spec.body(self.cfg.parent_body)
    else:
      parent = scene_spec.worldbody

    proj = (
      mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
      if self.cfg.orthographic
      else mujoco.mjtProjection.mjPROJ_PERSPECTIVE
    )
    parent.add_camera(
      name=self.cfg.name,
      pos=self.cfg.pos,
      quat=self.cfg.quat,
      fovy=self.cfg.fovy if self.cfg.fovy is not None else _DEFAULT_CAM_FOVY,
      resolution=[self.cfg.width, self.cfg.height],
      proj=proj,
    )

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    del model, data, device

    try:
      cam = mj_model.camera(self._camera_name)
      self._camera_idx = cam.id
    except KeyError as e:
      available = [mj_model.camera(i).name for i in range(mj_model.ncam)]
      raise ValueError(
        f"Camera '{self._camera_name}' not found in model. Available: {available}"
      ) from e

  def set_context(self, ctx: SensorContext) -> None:
    self._ctx = ctx

  def _compute_data(self) -> CameraSensorData:
    if self._ctx is None:
      raise RuntimeError(
        "CameraSensor requires a SensorContext. "
        "Ensure the sensor is part of a scene with "
        "sim.sense() calls."
      )
    rgb_data = None
    depth_data = None
    seg_data = None
    if "rgb" in self.cfg.data_types:
      rgb = self._ctx.get_rgb(self._camera_idx)
      rgb_data = rgb.clone() if self.cfg.clone_data else rgb
    if "depth" in self.cfg.data_types:
      depth = self._ctx.get_depth(self._camera_idx)
      depth_data = depth.clone() if self.cfg.clone_data else depth
    if "segmentation" in self.cfg.data_types:
      seg = self._ctx.get_segmentation(self._camera_idx)
      seg_data = seg.clone() if self.cfg.clone_data else seg
    return CameraSensorData(rgb=rgb_data, depth=depth_data, segmentation=seg_data)
