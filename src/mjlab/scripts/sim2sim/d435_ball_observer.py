"""D435 RGB-D football observation for native MuJoCo sim-to-sim."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

FloatArray = npt.NDArray[np.float32]

D435_CAMERA_NAME = "sim_d435"
_FOOTBALL_TEXTURE_NAME = "sim_football_checker"
_FOOTBALL_MATERIAL_NAME = "sim_football_material"
_CAMERA_FRAME_TO_OPTICAL_FRAME = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class D435Config:
  """Simulated G1 D435 camera and football detector parameters."""

  width: int = 640
  height: int = 480
  horizontal_fov_deg: float = 87.0
  vertical_fov_deg: float = 58.0
  min_depth: float = 0.3
  max_depth: float = 5.0
  depth_roi_px: int = 4
  ball_radius: float = 0.1098
  confidence_threshold: float = 0.25
  iou_threshold: float = 0.5
  ball_class_id: int = 0
  max_hold_time: float = 0.5
  camera_name: str = D435_CAMERA_NAME
  camera_pos_pelvis: tuple[float, float, float] = (
    0.14764571478,
    0.0,
    0.4626817855,
  )
  camera_quat_pelvis_wxyz: tuple[float, float, float, float] = (
    0.69411524,
    0.13492234,
    -0.13492234,
    -0.69411524,
  )

  @property
  def intrinsics(self) -> tuple[float, float, float, float]:
    fx = 0.5 * self.width / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)
    fy = 0.5 * self.height / math.tan(math.radians(self.vertical_fov_deg) / 2.0)
    return fx, fy, 0.5 * (self.width - 1.0), 0.5 * (self.height - 1.0)


def default_yolo_model() -> Path:
  """Find the shared football YOLO model in the neighboring project directories."""
  project_root = Path(__file__).resolve().parents[4]
  candidates = (
    project_root.parent / "klavier_rl_lab/outputs/sim_data_det_0126.onnx",
    project_root.parent
    / "klavier_rl_deploy-isaacsim5.1/deploy/robots/g1/config/policy/"
    "velocity/football/exported/sim_data_det_0126.onnx",
  )
  for path in candidates:
    if path.is_file():
      return path
  raise FileNotFoundError(
    "Football YOLO model was not found. Pass --yolo-model explicitly."
  )


def add_d435_camera(spec: mujoco.MjSpec, cfg: D435Config) -> None:
  """Attach the simulated D435 camera to the compiled scene's robot pelvis."""
  pelvis = spec.body("robot/pelvis")
  if pelvis is None:
    raise ValueError("MJLab scene is missing body 'robot/pelvis'.")
  camera = pelvis.add_camera(
    name=cfg.camera_name,
    pos=cfg.camera_pos_pelvis,
    quat=cfg.camera_quat_pelvis_wxyz,
  )
  fx, fy, _, _ = cfg.intrinsics
  camera.focal_pixel = np.asarray((fx, fy), dtype=np.float64)
  camera.principal_pixel = np.zeros(2, dtype=np.float64)
  camera.resolution = np.asarray((cfg.width, cfg.height), dtype=np.int32)
  camera.sensor_size = np.asarray((cfg.width, cfg.height), dtype=np.float64)


def add_football_visual_material(spec: mujoco.MjSpec) -> None:
  """Give the sim2sim ball the black-white appearance used by the detector."""
  spec.add_texture(
    name=_FOOTBALL_TEXTURE_NAME,
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(1.0, 1.0, 1.0),
    rgb2=(0.02, 0.02, 0.02),
    width=256,
    height=256,
  )
  material = spec.add_material(
    name=_FOOTBALL_MATERIAL_NAME,
    texuniform=True,
    texrepeat=(25.0, 25.0),
    specular=0.25,
    shininess=0.35,
  )
  material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _FOOTBALL_TEXTURE_NAME
  ball_geom = spec.geom("ball/ball_collision")
  if ball_geom is None:
    raise ValueError("MJLab scene is missing geom 'ball/ball_collision'.")
  ball_geom.material = _FOOTBALL_MATERIAL_NAME
  ball_geom.group = 0


def _quat_to_matrix(quat_wxyz: npt.ArrayLike) -> npt.NDArray[np.float64]:
  quat = np.asarray(quat_wxyz, dtype=np.float64)
  quat /= np.linalg.norm(quat)
  w, x, y, z = quat
  return np.asarray(
    (
      (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
      (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
      (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
    ),
    dtype=np.float64,
  )


def world_to_yaw(vector_w: npt.ArrayLike, root_quat_wxyz: npt.ArrayLike) -> FloatArray:
  """Express a world-frame vector in the robot's gravity-aligned yaw frame."""
  w, x, y, z = np.asarray(root_quat_wxyz, dtype=np.float64)
  yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  cos_yaw = math.cos(yaw)
  sin_yaw = math.sin(yaw)
  rotation_w_yaw = np.asarray(
    ((cos_yaw, -sin_yaw, 0.0), (sin_yaw, cos_yaw, 0.0), (0.0, 0.0, 1.0))
  )
  return (rotation_w_yaw.T @ np.asarray(vector_w, dtype=np.float64)).astype(np.float32)


def camera_point_to_yaw(
  point_optical: npt.ArrayLike,
  root_quat_wxyz: npt.ArrayLike,
  cfg: D435Config,
) -> FloatArray:
  """Transform a D435 optical-frame point into the robot yaw-aligned frame."""
  camera_rotation_pelvis = _quat_to_matrix(cfg.camera_quat_pelvis_wxyz)
  optical_rotation_pelvis = camera_rotation_pelvis @ _CAMERA_FRAME_TO_OPTICAL_FRAME
  point_pelvis = np.asarray(cfg.camera_pos_pelvis) + (
    optical_rotation_pelvis @ np.asarray(point_optical, dtype=np.float64)
  )
  point_world = _quat_to_matrix(root_quat_wxyz) @ point_pelvis
  return world_to_yaw(point_world, root_quat_wxyz)


def _resize_bilinear(
  image: npt.NDArray[np.uint8], width: int, height: int
) -> npt.NDArray[np.uint8]:
  """Resize an RGB image without adding an OpenCV runtime dependency."""
  src_height, src_width = image.shape[:2]
  if (src_width, src_height) == (width, height):
    return image.copy()
  x = np.linspace(0.0, src_width - 1.0, width)
  y = np.linspace(0.0, src_height - 1.0, height)
  x0 = np.floor(x).astype(np.int32)
  y0 = np.floor(y).astype(np.int32)
  x1 = np.minimum(x0 + 1, src_width - 1)
  y1 = np.minimum(y0 + 1, src_height - 1)
  wx = (x - x0)[None, :, None]
  wy = (y - y0)[:, None, None]
  top = (
    image[y0[:, None], x0[None, :]] * (1.0 - wx) + image[y0[:, None], x1[None, :]] * wx
  )
  bottom = (
    image[y1[:, None], x0[None, :]] * (1.0 - wx) + image[y1[:, None], x1[None, :]] * wx
  )
  return np.clip(top * (1.0 - wy) + bottom * wy, 0.0, 255.0).astype(np.uint8)


class YoloBallDetector:
  """Minimal YOLO ONNX detector for the simulated D435 RGB stream."""

  def __init__(self, model_path: Path, cfg: D435Config) -> None:
    if not model_path.is_file():
      raise FileNotFoundError(f"Football YOLO model does not exist: {model_path}")
    self.cfg = cfg
    self.session = ort.InferenceSession(
      str(model_path), providers=["CPUExecutionProvider"]
    )
    inputs = self.session.get_inputs()
    if len(inputs) != 1:
      raise ValueError("Football YOLO model must have exactly one input.")
    self.input_name = inputs[0].name
    shape = inputs[0].shape
    if not isinstance(shape[-2], int) or not isinstance(shape[-1], int):
      raise ValueError("Football YOLO model must have a fixed image input size.")
    self.input_height = shape[-2]
    self.input_width = shape[-1]

  def _preprocess(
    self, rgb: npt.NDArray[np.uint8]
  ) -> tuple[FloatArray, float, float, float]:
    height, width = rgb.shape[:2]
    scale = min(self.input_width / width, self.input_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = _resize_bilinear(rgb, resized_width, resized_height)
    pad_x = 0.5 * (self.input_width - resized_width)
    pad_y = 0.5 * (self.input_height - resized_height)
    left = int(math.floor(pad_x))
    top = int(math.floor(pad_y))
    canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, scale, float(left), float(top)

  def detect(self, rgb: npt.NDArray[np.uint8]) -> tuple[FloatArray, float] | None:
    """Return the best ball bounding box in source pixels and its confidence."""
    tensor, scale, pad_x, pad_y = self._preprocess(rgb)
    output = np.asarray(
      self.session.run(None, {self.input_name: tensor})[0], dtype=np.float32
    ).squeeze()
    if output.ndim != 2:
      raise ValueError(f"Unexpected YOLO output shape: {output.shape}")
    if output.shape[0] < output.shape[1]:
      output = output.T
    if output.shape[1] < 5:
      raise ValueError(f"Unexpected YOLO prediction shape: {output.shape}")

    boxes_xywh = output[:, :4]
    class_scores = output[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = (class_ids == self.cfg.ball_class_id) & (
      scores >= self.cfg.confidence_threshold
    )
    if not np.any(keep):
      return None

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - 0.5 * boxes_xywh[:, 2]
    boxes[:, 1] = boxes_xywh[:, 1] - 0.5 * boxes_xywh[:, 3]
    boxes[:, 2] = boxes_xywh[:, 0] + 0.5 * boxes_xywh[:, 2]
    boxes[:, 3] = boxes_xywh[:, 1] + 0.5 * boxes_xywh[:, 3]
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - pad_x) / scale
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - pad_y) / scale
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0.0, rgb.shape[1] - 1.0)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0.0, rgb.shape[0] - 1.0)

    # Class filtering leaves only football candidates. Standard NMS always keeps
    # the highest-confidence candidate, which is the only result needed here.
    best = int(np.argmax(scores))
    return boxes[best], float(scores[best])


class D435BallObserver:
  """Estimate yaw-frame football observations from simulated D435 RGB-D images."""

  def __init__(
    self,
    model: mujoco.MjModel,
    *,
    root_body_id: int,
    foot_body_ids: npt.NDArray[np.int32],
    yolo_model: Path,
    cfg: D435Config,
  ) -> None:
    self.cfg = cfg
    self.root_body_id = root_body_id
    self.foot_body_ids = foot_body_ids
    self.renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
    self.scene_option = mujoco.MjvOption()
    self.detector = YoloBallDetector(yolo_model, cfg)
    self.last_ball_pos_yaw: FloatArray | None = None
    self.last_detection_time = -math.inf
    self.last_rgb: npt.NDArray[np.uint8] | None = None
    self.last_depth: FloatArray | None = None
    self.last_detection: tuple[FloatArray, float] | None = None

  def reset(self) -> None:
    self.last_ball_pos_yaw = None
    self.last_detection_time = -math.inf
    self.last_detection = None

  def close(self) -> None:
    self.renderer.close()

  def _render(self, data: mujoco.MjData) -> tuple[npt.NDArray[np.uint8], FloatArray]:
    self.renderer.disable_depth_rendering()
    self.renderer.update_scene(
      data, camera=self.cfg.camera_name, scene_option=self.scene_option
    )
    rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
    self.renderer.enable_depth_rendering()
    self.renderer.update_scene(
      data, camera=self.cfg.camera_name, scene_option=self.scene_option
    )
    depth = np.asarray(self.renderer.render(), dtype=np.float32).copy()
    self.renderer.disable_depth_rendering()
    self.last_rgb = rgb
    self.last_depth = depth
    return rgb, depth

  def _depth_at_box(
    self, depth: FloatArray, box: FloatArray
  ) -> tuple[int, int, float] | None:
    center_u = int(round(0.5 * (float(box[0]) + float(box[2]))))
    center_v = int(round(0.5 * (float(box[1]) + float(box[3]))))
    half = max(1, self.cfg.depth_roi_px)
    u1, u2 = max(0, center_u - half), min(depth.shape[1], center_u + half + 1)
    v1, v2 = max(0, center_v - half), min(depth.shape[0], center_v + half + 1)
    values = depth[v1:v2, u1:u2]
    values = values[
      np.isfinite(values)
      & (values >= self.cfg.min_depth)
      & (values <= self.cfg.max_depth)
    ]
    if values.size == 0:
      return None
    return center_u, center_v, float(np.median(values))

  def observe(self, data: mujoco.MjData) -> tuple[FloatArray, FloatArray]:
    rgb, depth = self._render(data)
    detection = self.detector.detect(rgb)
    self.last_detection = detection
    root_pos = data.xpos[self.root_body_id]
    root_quat = data.xquat[self.root_body_id]

    if detection is not None:
      box, _ = detection
      depth_sample = self._depth_at_box(depth, box)
      if depth_sample is not None:
        u, v, distance = depth_sample
        fx, fy, cx, cy = self.cfg.intrinsics
        point_optical = np.asarray(
          ((u - cx) * distance / fx, (v - cy) * distance / fy, distance)
        )
        ray_norm = np.linalg.norm(point_optical)
        if ray_norm > 1e-6:
          point_optical += point_optical / ray_norm * self.cfg.ball_radius
        self.last_ball_pos_yaw = camera_point_to_yaw(point_optical, root_quat, self.cfg)
        self.last_detection_time = float(data.time)

    if (
      self.last_ball_pos_yaw is None
      or float(data.time) - self.last_detection_time > self.cfg.max_hold_time
    ):
      return np.zeros(2, dtype=np.float32), np.zeros(4, dtype=np.float32)

    feet_relative_w = data.xpos[self.foot_body_ids] - root_pos
    feet_yaw = np.stack(
      [world_to_yaw(position, root_quat) for position in feet_relative_w]
    )
    foot_to_ball = self.last_ball_pos_yaw[None, :2] - feet_yaw[:, :2]
    return (
      self.last_ball_pos_yaw[:2].copy(),
      foot_to_ball.astype(np.float32).reshape(-1),
    )


class MujocoBallObserver:
  """Ground-truth observer retained for debugging the visual pipeline."""

  def __init__(
    self, root_body_id: int, ball_body_id: int, foot_body_ids: npt.NDArray[np.int32]
  ) -> None:
    self.root_body_id = root_body_id
    self.ball_body_id = ball_body_id
    self.foot_body_ids = foot_body_ids

  def reset(self) -> None:
    pass

  def close(self) -> None:
    pass

  def observe(self, data: mujoco.MjData) -> tuple[FloatArray, FloatArray]:
    root_pos = data.xpos[self.root_body_id]
    root_quat = data.xquat[self.root_body_id]
    ball_yaw = world_to_yaw(data.xpos[self.ball_body_id] - root_pos, root_quat)
    feet_yaw = np.stack(
      [
        world_to_yaw(position - root_pos, root_quat)
        for position in data.xpos[self.foot_body_ids]
      ]
    )
    foot_to_ball = ball_yaw[None, :2] - feet_yaw[:, :2]
    return ball_yaw[:2], foot_to_ball.astype(np.float32).reshape(-1)


def make_ball_observer(
  source: str,
  model: mujoco.MjModel,
  *,
  root_body_id: int,
  ball_body_id: int,
  foot_body_ids: npt.NDArray[np.int32],
  yolo_model: Path | None,
  cfg: D435Config,
) -> Any:
  """Build either the deployment-like D435 observer or a truth debug observer."""
  if source == "d435":
    return D435BallObserver(
      model,
      root_body_id=root_body_id,
      foot_body_ids=foot_body_ids,
      yolo_model=yolo_model or default_yolo_model(),
      cfg=cfg,
    )
  if source == "mujoco":
    return MujocoBallObserver(root_body_id, ball_body_id, foot_body_ids)
  raise ValueError(f"Unsupported ball observation source: {source!r}")
