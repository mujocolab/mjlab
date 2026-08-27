"""D435 RGB-D football observation for native MuJoCo sim-to-sim."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

FloatArray = npt.NDArray[np.float32]

D435_CAMERA_NAME = "sim_d435"
_FOOTBALL_TEXTURE_NAME = "sim_football_checker"
_FOOTBALL_MATERIAL_NAME = "sim_football_material"
_OPTICAL_TO_CAMERA_FRAME = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class D435Config:
  """Python mirror of the production football RGB-D configuration."""

  width: int = 640
  height: int = 480
  rgb_fovy_deg: float = 42.5
  depth_fovy_deg: float = 58.0
  min_depth: float = 0.3
  max_depth: float = 3.0
  depth_roi_px: int = 4
  ball_radius: float = 0.1098
  confidence_threshold: float = 0.5
  iou_threshold: float = 0.5
  ball_class_id: int = 0
  max_detections: int = 5
  max_nms_candidates: int = 3000
  max_hold_time: float = 0.5
  vision_mode: Literal["deployment_rgbd", "robocup"] = "deployment_rgbd"
  ground_height: float = 0.0
  camera_name: str = D435_CAMERA_NAME
  # torso -> camera pose: Unitree's official factory D435i mount, taken from
  # the stock (unmodified) g1_29dof.xml / g1_23dof.xml MJCF description
  # (matches the C++ deployment fallback default in football_observations.h).
  # This project had been using a custom-remounted position instead
  # ([0.1135993074, 0.01753, 0.3934754688]).
  camera_pos_torso: tuple[float, float, float] = (
    0.0576235,
    0.01753,
    0.42987,
  )
  camera_quat_torso_wxyz: tuple[float, float, float, float] = (
    0.6592524821,
    0.2557071857,
    -0.2557071857,
    -0.6592524821,
  )

  @property
  def intrinsics(self) -> tuple[float, float, float, float]:
    # The deployment deprojects aligned depth with RGB intrinsics. Its FOV
    # fallback derives both focal lengths from the vertical RGB FOV.
    focal = 0.5 * self.height / math.tan(math.radians(self.rgb_fovy_deg) / 2.0)
    return focal, focal, 0.5 * self.width, 0.5 * self.height


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
  """Attach the RGB camera exactly as in the deployment football XML."""
  torso = spec.body("robot/torso_link")
  if torso is None:
    raise ValueError("MJLab scene is missing body 'robot/torso_link'.")
  camera = torso.add_camera(
    name=cfg.camera_name,
    pos=cfg.camera_pos_torso,
    quat=cfg.camera_quat_torso_wxyz,
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
  camera_pos_w: npt.ArrayLike,
  camera_rotation_w: npt.ArrayLike,
  root_pos_w: npt.ArrayLike,
  root_quat_wxyz: npt.ArrayLike,
) -> FloatArray:
  """Match deployment optical -> camera -> world -> robot-yaw conversion."""
  point_camera = _OPTICAL_TO_CAMERA_FRAME @ np.asarray(point_optical, dtype=np.float64)
  point_world = np.asarray(camera_pos_w, dtype=np.float64) + (
    np.asarray(camera_rotation_w, dtype=np.float64).reshape(3, 3) @ point_camera
  )
  return world_to_yaw(
    point_world - np.asarray(root_pos_w, dtype=np.float64), root_quat_wxyz
  )


class YoloBallDetector:
  """Python port of the deployment YoloBallDetector ONNX path."""

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
    resized_width = max(1, int(math.floor(width * scale + 0.5)))
    resized_height = max(1, int(math.floor(height * scale + 0.5)))
    pad_width = self.input_width - resized_width
    pad_height = self.input_height - resized_height
    if self.cfg.vision_mode == "robocup":
      pad_left = 0
      pad_top = 0
      pad_value = 0.0
    else:
      pad_left = int(math.floor(pad_width / 2.0 - 0.1 + 0.5))
      pad_top = int(math.floor(pad_height / 2.0 - 0.1 + 0.5))
      pad_value = 114.0

    # Match the deployment's half-pixel bilinear sampler, including leaving
    # target pixels outside the valid source interval at letterbox gray.
    target_x = np.arange(self.input_width, dtype=np.float32)
    target_y = np.arange(self.input_height, dtype=np.float32)
    source_x = (target_x - pad_left + 0.5) / scale - 0.5
    source_y = (target_y - pad_top + 0.5) / scale - 0.5
    valid_x = (source_x >= 0.0) & (source_x <= width - 1)
    valid_y = (source_y >= 0.0) & (source_y <= height - 1)
    x0 = np.clip(np.floor(source_x).astype(np.int32), 0, width - 1)
    y0 = np.clip(np.floor(source_y).astype(np.int32), 0, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    weight_x = (source_x - x0)[None, :, None]
    weight_y = (source_y - y0)[:, None, None]
    top = (
      rgb[y0[:, None], x0[None, :]] * (1.0 - weight_x)
      + rgb[y0[:, None], x1[None, :]] * weight_x
    )
    bottom = (
      rgb[y1[:, None], x0[None, :]] * (1.0 - weight_x)
      + rgb[y1[:, None], x1[None, :]] * weight_x
    )
    sampled = top * (1.0 - weight_y) + bottom * weight_y
    canvas = np.full(
      (self.input_height, self.input_width, 3), pad_value, dtype=np.float32
    )
    valid = valid_y[:, None] & valid_x[None, :]
    canvas[valid] = sampled[valid]
    tensor = canvas.transpose(2, 0, 1)[None] / 255.0
    return tensor.astype(np.float32), scale, float(pad_left), float(pad_top)

  @staticmethod
  def _box_iou(left: FloatArray, right: FloatArray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(
      0.0, float(right[3] - right[1])
    )
    return intersection / max(left_area + right_area - intersection, 1.0e-6)

  def _decode_output(
    self,
    output: Any,
    image_width: int,
    image_height: int,
    scale: float,
    pad_x: float,
    pad_y: float,
  ) -> list[tuple[FloatArray, float]]:
    predictions = np.asarray(output, dtype=np.float32)
    while predictions.ndim > 2 and predictions.shape[0] == 1:
      predictions = predictions[0]
    if predictions.ndim != 2:
      return []
    rows, cols = predictions.shape
    if (rows in (6, 7, 84, 85) or rows < cols) and rows >= 6:
      predictions = predictions.T
      rows, cols = predictions.shape
    if rows <= 0 or cols < 6:
      return []

    candidates: list[tuple[FloatArray, float]] = []
    for prediction in predictions:
      if cols in (6, 7):
        offset = 1 if cols == 7 else 0
        box = prediction[offset : offset + 4].copy()
        confidence = float(prediction[offset + 4])
        class_id = int(math.floor(float(prediction[offset + 5]) + 0.5))
      else:
        center_x, center_y, width, height = (float(value) for value in prediction[:4])
        box = np.asarray(
          (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
          ),
          dtype=np.float32,
        )
        class_start = 5 if cols == 85 else 4
        objectness = float(prediction[4]) if cols == 85 else 1.0
        class_scores = objectness * prediction[class_start:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

      if (
        class_id != self.cfg.ball_class_id or confidence < self.cfg.confidence_threshold
      ):
        continue
      box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
      box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
      box[[0, 2]] = np.clip(box[[0, 2]], 0.0, image_width - 1.0)
      box[[1, 3]] = np.clip(box[[1, 3]], 0.0, image_height - 1.0)
      if box[2] <= box[0] or box[3] <= box[1]:
        continue
      candidates.append((box, confidence))

    candidates.sort(key=lambda item: item[1], reverse=True)
    candidates = candidates[: self.cfg.max_nms_candidates]
    kept: list[tuple[FloatArray, float]] = []
    for candidate in candidates:
      if any(
        self._box_iou(candidate[0], previous[0]) > self.cfg.iou_threshold
        for previous in kept
      ):
        continue
      kept.append(candidate)
      if len(kept) >= self.cfg.max_detections:
        break
    return kept

  def detect(self, rgb: npt.NDArray[np.uint8]) -> tuple[FloatArray, float] | None:
    """Return the deployment-selected ball box and confidence."""
    tensor, scale, pad_x, pad_y = self._preprocess(rgb)
    detections: list[tuple[FloatArray, float]] = []
    for output in self.session.run(None, {self.input_name: tensor}):
      detections.extend(
        self._decode_output(
          output,
          rgb.shape[1],
          rgb.shape[0],
          scale,
          pad_x,
          pad_y,
        )
      )
    if not detections:
      return None

    if self.cfg.vision_mode == "robocup":
      return max(detections, key=lambda item: item[1])

    image_area = float(rgb.shape[0] * rgb.shape[1])
    return max(
      detections,
      key=lambda item: item[1]
      / (1.0 + 8.0 * max(float(np.prod(item[0][2:] - item[0][:2])), 1.0) / image_area),
    )


def project_bbox_bottom_to_ground_yaw(
  box: npt.ArrayLike,
  intrinsics: tuple[float, float, float, float],
  camera_pos_w: npt.ArrayLike,
  camera_rotation_w: npt.ArrayLike,
  root_pos_w: npt.ArrayLike,
  root_quat_wxyz: npt.ArrayLike,
  *,
  ground_height: float = 0.0,
) -> FloatArray | None:
  """Port RoboCup's bbox-bottom optical-ray/ground-plane intersection."""
  x1, _, x2, y2 = (float(value) for value in np.asarray(box).reshape(4))
  fx, fy, cx, cy = intrinsics
  pixel_u = 0.5 * (x1 + x2)
  ray_optical = np.asarray(((pixel_u - cx) / fx, (y2 - cy) / fy, 1.0), dtype=np.float64)
  ray_camera = _OPTICAL_TO_CAMERA_FRAME @ ray_optical
  ray_world = np.asarray(camera_rotation_w, dtype=np.float64).reshape(3, 3) @ ray_camera
  camera_pos = np.asarray(camera_pos_w, dtype=np.float64)
  if abs(float(ray_world[2])) < 1.0e-8:
    return None
  scale = (ground_height - float(camera_pos[2])) / float(ray_world[2])
  if not math.isfinite(scale) or scale <= 0.0:
    return None
  point_world = camera_pos + scale * ray_world
  return world_to_yaw(
    point_world - np.asarray(root_pos_w, dtype=np.float64), root_quat_wxyz
  )


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
    self.camera_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.camera_name
    )
    if self.camera_id < 0:
      raise ValueError(f"MuJoCo model is missing camera {cfg.camera_name!r}.")
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
    if self.cfg.vision_mode == "robocup":
      self.last_rgb = rgb
      self.last_depth = None
      return rgb, np.empty((0, 0), dtype=np.float32)
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
    center_u = int(math.floor(0.5 * (float(box[0]) + float(box[2])) + 0.5))
    center_v = int(math.floor(0.5 * (float(box[1]) + float(box[3])) + 0.5))
    half = max(1, self.cfg.depth_roi_px)
    u1, u2 = max(0, center_u - half), min(depth.shape[1], center_u + half + 1)
    v1, v2 = max(0, center_v - half), min(depth.shape[0], center_v + half + 1)
    distance = self._median_valid_depth(depth[v1:v2, u1:u2])
    if distance is None:
      # The synchronized deployment falls back from the center ROI to the
      # complete aligned RGB bounding box.
      x1 = max(0, int(math.floor(float(box[0]))))
      y1 = max(0, int(math.floor(float(box[1]))))
      x2 = min(depth.shape[1], int(math.ceil(float(box[2]))))
      y2 = min(depth.shape[0], int(math.ceil(float(box[3]))))
      distance = self._median_valid_depth(depth[y1:y2, x1:x2])
    if distance is None:
      return None
    return center_u, center_v, distance

  def _median_valid_depth(self, values: FloatArray) -> float | None:
    valid = values[
      np.isfinite(values)
      & (values >= self.cfg.min_depth)
      & (values <= self.cfg.max_depth)
    ].reshape(-1)
    if valid.size == 0:
      return None
    # std::nth_element in deployment selects the upper middle sample rather
    # than averaging the two middle values for an even-sized region.
    middle = valid.size // 2
    return float(np.partition(valid, middle)[middle])

  def observe(self, data: mujoco.MjData) -> tuple[FloatArray, FloatArray]:
    rgb, depth = self._render(data)
    detection = self.detector.detect(rgb)
    self.last_detection = detection
    root_pos = data.xpos[self.root_body_id]
    root_quat = data.xquat[self.root_body_id]

    if detection is not None:
      box, _ = detection
      camera_rotation_w = data.cam_xmat[self.camera_id].reshape(3, 3)
      if self.cfg.vision_mode == "robocup":
        ball_pos_yaw = project_bbox_bottom_to_ground_yaw(
          box,
          self.cfg.intrinsics,
          data.cam_xpos[self.camera_id],
          camera_rotation_w,
          root_pos,
          root_quat,
          ground_height=self.cfg.ground_height,
        )
        if ball_pos_yaw is not None:
          self.last_ball_pos_yaw = ball_pos_yaw
          self.last_detection_time = float(data.time)
      else:
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
          self.last_ball_pos_yaw = camera_point_to_yaw(
            point_optical,
            data.cam_xpos[self.camera_id],
            camera_rotation_w,
            root_pos,
            root_quat,
          )
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
  if source in {"d435", "robocup"}:
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
