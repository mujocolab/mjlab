"""Run the depth-image G1 football Student in native CPU MuJoCo.

Unlike ``g1_football.py`` (coordinate policy, D435 RGB + YOLO ball detector),
this Student never sees an explicit ball position: it consumes a raw depth
image history directly, exactly matching ``DepthTemporalLatentStudentModel`` /
``DepthCoordinateStudentModel`` from ``mjlab.tasks.velocity_football_depth``.
The depth camera is not hand-added like the D435 RGB camera; it is already
part of the task's ``env_cfg.scene.sensors`` and is created automatically
when the scene is compiled, so its pose/fovy/resolution/geom-group filter
always match training by construction.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import tyro

import mjlab
import mjlab.tasks  # noqa: F401
from mjlab.scene import Scene
from mjlab.scripts.sim2sim.g1_football import (
  EXPECTED_ACTION_DIM,
  FRAME_STACK,
  PROPRIOCEPTIVE_OBS_DIM,
  PROPRIOCEPTIVE_OBSERVATION_NAMES,
  KeyboardController,
  _HistoryBuffer,
  _parse_csv,
  _parse_float_csv,
  _phase,
  configure_tracking_camera,
  quat_apply_inverse,
)
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity_football_depth import DEPTH_CANDIDATE_TASK_ID
from mjlab.tasks.velocity_football_depth.env_cfg import (
  DEPTH_MAX_METERS,
  DEPTH_MIN_METERS,
  DEPTH_SENSOR_NAME,
)

TASK_ID = DEPTH_CANDIDATE_TASK_ID

# Keep these deployment-only limits in lockstep with
# klavier_rl_deploy-isaacsim5.1_de/deploy/robots/g1/config/policy/velocity/
# football_mjlab_depth/params/deploy.yaml.  The policy was trained on the
# wider training range, while the joystick path intentionally exposes this
# smaller envelope.
DEPLOYED_COMMAND_MIN = np.asarray([-0.25, -0.25, -1.0], dtype=np.float32)
DEPLOYED_COMMAND_MAX = np.asarray([1.0, 0.25, 1.0], dtype=np.float32)

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class DepthPolicyMetadata:
  """Deployment parameters embedded in an exported depth-Student ONNX policy."""

  joint_names: tuple[str, ...]
  default_joint_pos: FloatArray
  action_scale: FloatArray
  depth_history_length: int
  depth_height: int
  depth_width: int

  @classmethod
  def from_session(cls, session: Any) -> DepthPolicyMetadata:
    """Parse and validate the two-input (proprio, depth) deployment contract."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_names = tuple(item.name for item in inputs)
    if input_names != ("proprio", "depth"):
      raise ValueError(
        f"Depth policy inputs must be ('proprio', 'depth'), got {input_names}."
      )
    if len(outputs) != 1 or outputs[0].name != "actions":
      raise ValueError("Policy must expose exactly one ONNX output named 'actions'.")

    proprio_dim = inputs[0].shape[-1]
    if proprio_dim != PROPRIOCEPTIVE_OBS_DIM:
      raise ValueError(
        f"Policy expects {proprio_dim} proprio inputs; this task requires "
        f"{PROPRIOCEPTIVE_OBS_DIM}."
      )

    depth_shape = tuple(inputs[1].shape)
    if len(depth_shape) != 4:
      raise ValueError(
        "Depth input must have shape (batch, history, height, width), got "
        f"{depth_shape}."
      )
    _, depth_history_length, depth_height, depth_width = depth_shape
    if not all(
      isinstance(v, int) and v > 0
      for v in (depth_history_length, depth_height, depth_width)
    ):
      raise ValueError(
        "Depth policy must expose a fixed positive (history, height, width), "
        f"got {depth_shape}."
      )
    if outputs[0].shape[-1] != EXPECTED_ACTION_DIM:
      raise ValueError(
        f"Policy produces {outputs[0].shape[-1]} actions; this task requires "
        f"{EXPECTED_ACTION_DIM}."
      )

    metadata = session.get_modelmeta().custom_metadata_map
    joint_names = _parse_csv(metadata, "joint_names")
    default_joint_pos = _parse_float_csv(metadata, "default_joint_pos")
    action_scale = _parse_float_csv(metadata, "action_scale")
    observation_names = _parse_csv(metadata, "observation_names")
    history_raw = _parse_float_csv(metadata, "observation_terms_history_length")
    meta_depth_dims = (
      int(float(metadata.get("depth_history_length", depth_history_length))),
      int(float(metadata.get("depth_height", depth_height))),
      int(float(metadata.get("depth_width", depth_width))),
    )

    if len(joint_names) != EXPECTED_ACTION_DIM:
      raise ValueError(
        f"Policy metadata contains {len(joint_names)} joints; expected "
        f"{EXPECTED_ACTION_DIM}."
      )
    if default_joint_pos.shape != (EXPECTED_ACTION_DIM,):
      raise ValueError("ONNX default_joint_pos must contain 29 values.")
    if action_scale.shape != (EXPECTED_ACTION_DIM,):
      raise ValueError("ONNX action_scale must contain 29 values.")
    if observation_names != PROPRIOCEPTIVE_OBSERVATION_NAMES:
      raise ValueError(
        "ONNX observation order does not match the depth Student contract: "
        f"{observation_names}."
      )
    if not np.all(history_raw == FRAME_STACK):
      raise ValueError(
        "ONNX observation history metadata does not match its input contract."
      )
    if meta_depth_dims != (depth_history_length, depth_height, depth_width):
      raise ValueError(
        f"ONNX depth metadata {meta_depth_dims} disagrees with the depth "
        f"input tensor shape {(depth_history_length, depth_height, depth_width)}."
      )

    return cls(
      joint_names=joint_names,
      default_joint_pos=default_joint_pos,
      action_scale=action_scale,
      depth_history_length=depth_history_length,
      depth_height=depth_height,
      depth_width=depth_width,
    )


class _DepthHistoryBuffer:
  """Chronological depth-frame history stored oldest to newest."""

  def __init__(self, length: int) -> None:
    self._length = length
    self._frames: list[FloatArray] = []

  def reset(self, frame: FloatArray) -> None:
    self._frames = [frame.copy() for _ in range(self._length)]

  def append(self, frame: FloatArray) -> None:
    if not self._frames:
      self.reset(frame)
      return
    self._frames.append(frame.copy())
    self._frames.pop(0)

  def stacked(self) -> FloatArray:
    if not self._frames:
      raise RuntimeError("Depth history has not been initialized.")
    return np.stack(self._frames, axis=0).astype(np.float32, copy=False)


class DepthLatencyQueue:
  """Delay depth frames before they reach the policy, oldest-first.

  Sim2sim renders depth synchronously every policy step, so it is always
  perfectly fresh. Real deployment isn't: the camera capture and inference
  threads run independently of the 50Hz control loop (camera at its own
  ~60Hz), and the control thread just reads whatever the camera thread most
  recently finished (see ``detail::read_camera_frame`` in the deploy repo's
  ``observations.h`` and ``AsyncFootballVisionRuntime``), which is
  occasionally a step or more stale relative to "right now". This models
  that: a fixed base delay plus optional per-step random jitter, so the
  frame handed to the policy this step was actually rendered
  ``base_delay_steps`` to ``base_delay_steps + jitter_steps`` policy steps
  ago instead of this one.
  """

  def __init__(
    self, base_delay_steps: int, jitter_steps: int, rng: np.random.Generator
  ) -> None:
    if base_delay_steps < 0:
      raise ValueError(f"base_delay_steps must be non-negative, got {base_delay_steps}")
    if jitter_steps < 0:
      raise ValueError(f"jitter_steps must be non-negative, got {jitter_steps}")
    self._base_delay = base_delay_steps
    self._jitter = jitter_steps
    self._rng = rng
    self._buffer: list[FloatArray] = []

  @property
  def _capacity(self) -> int:
    return self._base_delay + self._jitter + 1

  def reset(self, frame: FloatArray) -> FloatArray:
    self._buffer = [frame.copy() for _ in range(self._capacity)]
    return frame.copy()

  def push(self, frame: FloatArray) -> FloatArray:
    self._buffer.append(frame.copy())
    if len(self._buffer) > self._capacity:
      self._buffer.pop(0)
    delay = self._base_delay
    if self._jitter > 0:
      delay += int(self._rng.integers(0, self._jitter + 1))
    index = max(0, len(self._buffer) - 1 - delay)
    return self._buffer[index]


class DepthWindow:
  """Show the depth Student's own eyes: the latest normalized policy frame.

  The native MuJoCo passive viewer has no API for compositing an arbitrary 2D
  image into its 3D window, so this mirrors ``DetectionWindow`` and opens a
  small separate matplotlib figure updated at a throttled rate.
  """

  def __init__(
    self,
    height: int,
    width: int,
    min_depth: float,
    max_depth: float,
    update_rate: float,
  ) -> None:
    if update_rate <= 0.0:
      raise ValueError("Depth window update rate must be positive.")

    # Importing pyplot selects a GUI backend, so defer it until the optional
    # interactive window is actually requested.
    from matplotlib import pyplot as plt

    self._plt = plt
    self._update_period = 1.0 / update_rate
    self._last_update_time = -math.inf
    self._figure, self._axis = plt.subplots(
      1, 1, figsize=(4.8, 4.0), num="Depth Student view"
    )
    self._image = self._axis.imshow(
      np.ones((height, width), dtype=np.float32),
      cmap="viridis",
      vmin=0.0,
      vmax=1.0,
    )
    self._axis.set_title("Policy depth input (latest frame)")
    self._axis.set_axis_off()
    colorbar = self._figure.colorbar(self._image, ax=self._axis, fraction=0.046)
    colorbar.set_label(f"0={min_depth:.2f} m, 1={max_depth:.2f} m or invalid")
    self._figure.tight_layout()
    plt.show(block=False)

  @property
  def is_open(self) -> bool:
    """Return whether the user has kept the diagnostic figure open."""
    return bool(self._plt.fignum_exists(self._figure.number))

  def update(self, depth_frame: FloatArray, sim_time: float) -> None:
    """Refresh the displayed frame, throttled to the configured rate."""
    if not self.is_open or sim_time - self._last_update_time < self._update_period:
      return
    self._last_update_time = sim_time
    self._image.set_data(depth_frame)
    self._figure.canvas.draw_idle()
    self._figure.canvas.flush_events()

  def close(self) -> None:
    """Close the diagnostic figure if it is still open."""
    if self.is_open:
      self._plt.close(self._figure)


class ProprioAssembler:
  """Reproduce the Student's five-frame proprioceptive history, term-major."""

  def __init__(self) -> None:
    self._history = {
      name: _HistoryBuffer(FRAME_STACK) for name in PROPRIOCEPTIVE_OBSERVATION_NAMES
    }

  def _validate(self, terms: dict[str, FloatArray]) -> None:
    if tuple(terms) != PROPRIOCEPTIVE_OBSERVATION_NAMES:
      raise ValueError(
        f"Observation terms must be ordered as {PROPRIOCEPTIVE_OBSERVATION_NAMES}."
      )

  def reset(self, terms: dict[str, FloatArray]) -> FloatArray:
    self._validate(terms)
    for name, value in terms.items():
      self._history[name].reset(value)
    return self.observation()

  def append(self, terms: dict[str, FloatArray]) -> FloatArray:
    self._validate(terms)
    for name, value in terms.items():
      self._history[name].append(value)
    return self.observation()

  def observation(self) -> FloatArray:
    obs = np.concatenate(
      [
        self._history[name].flatten(FRAME_STACK)
        for name in PROPRIOCEPTIVE_OBSERVATION_NAMES
      ]
    ).astype(np.float32, copy=False)
    if obs.shape != (PROPRIOCEPTIVE_OBS_DIM,):
      raise RuntimeError(f"Assembled invalid proprio observation shape {obs.shape}.")
    if not np.all(np.isfinite(obs)):
      raise RuntimeError("Sim-to-sim proprio observation contains NaN or Inf values.")
    return obs


def normalize_and_downsample_depth(
  depth_raw: FloatArray,
  *,
  min_depth: float,
  max_depth: float,
  out_height: int,
  out_width: int,
) -> FloatArray:
  """Reproduce ``normalized_camera_depth`` + ``normalized_camera_depth_frame``.

  Invalid pixels (non-finite or outside [min_depth, max_depth]) are mapped to
  one, matching training's convention of treating missing depth as "far"
  rather than "an obstacle at the camera origin". Downsampling uses an exact
  block average, equivalent to ``F.interpolate(mode="area")`` for integer
  downsampling factors.
  """
  valid = np.isfinite(depth_raw) & (depth_raw >= min_depth) & (depth_raw <= max_depth)
  normalized = np.clip(depth_raw, min_depth, max_depth) / max_depth
  normalized = np.where(valid, normalized, 1.0).astype(np.float32)

  height, width = normalized.shape
  if (height, width) == (out_height, out_width):
    return normalized
  if height % out_height != 0 or width % out_width != 0:
    raise ValueError(
      f"Raw depth resolution {(height, width)} is not an integer multiple of "
      f"the policy resolution {(out_height, out_width)}."
    )
  factor_h, factor_w = height // out_height, width // out_width
  return (
    normalized.reshape(out_height, factor_h, out_width, factor_w)
    .mean(axis=(1, 3))
    .astype(np.float32)
  )


@dataclass(frozen=True)
class ModelBindings:
  """Name-based indices connecting policy order to a compiled MuJoCo model."""

  joint_qpos_adr: npt.NDArray[np.int32]
  joint_dof_adr: npt.NDArray[np.int32]
  actuator_ids: npt.NDArray[np.int32]
  joint_limits: FloatArray
  root_body_id: int
  imu_sensor_adr: int
  depth_camera_id: int
  init_key_id: int

  @classmethod
  def from_model(
    cls, model: mujoco.MjModel, joint_names: tuple[str, ...]
  ) -> ModelBindings:
    """Resolve model addresses by stable names and fail on missing elements."""

    def require_id(obj_type: mujoco.mjtObj, name: str) -> int:
      obj_id = mujoco.mj_name2id(model, obj_type, name)
      if obj_id < 0:
        raise ValueError(f"MuJoCo model is missing required object {name!r}.")
      return obj_id

    joint_ids = np.asarray(
      [require_id(mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}") for name in joint_names],
      dtype=np.int32,
    )
    actuator_ids = np.asarray(
      [
        require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{name}")
        for name in joint_names
      ],
      dtype=np.int32,
    )
    imu_sensor_id = require_id(mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")
    if model.sensor_dim[imu_sensor_id] != 3:
      raise ValueError("robot/imu_ang_vel must be a three-dimensional sensor.")
    if not np.all(model.jnt_limited[joint_ids]):
      raise ValueError("Every policy joint must have a finite MuJoCo position limit.")
    return cls(
      joint_qpos_adr=model.jnt_qposadr[joint_ids].astype(np.int32),
      joint_dof_adr=model.jnt_dofadr[joint_ids].astype(np.int32),
      actuator_ids=actuator_ids,
      joint_limits=model.jnt_range[joint_ids].astype(np.float32),
      root_body_id=require_id(mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis"),
      imu_sensor_adr=int(model.sensor_adr[imu_sensor_id]),
      depth_camera_id=require_id(mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_SENSOR_NAME),
      init_key_id=require_id(mujoco.mjtObj.mjOBJ_KEY, "init_state"),
    )


class TrainingActionProcessor:
  """Apply actions exactly like the training JointPositionAction."""

  def __init__(self, metadata: DepthPolicyMetadata) -> None:
    self._metadata = metadata

  def reset(self) -> FloatArray:
    return self._metadata.default_joint_pos.copy()

  def process(self, raw_action: FloatArray) -> FloatArray:
    action = np.asarray(raw_action, dtype=np.float32)
    if action.shape != (EXPECTED_ACTION_DIM,):
      raise RuntimeError(f"Policy returned invalid action shape {action.shape}.")
    if not np.all(np.isfinite(action)):
      raise RuntimeError("Policy returned NaN or Inf actions.")
    return (
      self._metadata.default_joint_pos + self._metadata.action_scale * action
    ).astype(np.float32)


@dataclass
class CameraPositionResetRandomizer:
  """Resample one fixed camera translation around calibration at every reset."""

  model: mujoco.MjModel
  camera_id: int
  nominal_position: FloatArray
  jitter_meters: float
  rng: np.random.Generator

  def reset(self) -> FloatArray:
    offset = self.rng.uniform(
      -self.jitter_meters,
      self.jitter_meters,
      size=3,
    ).astype(np.float32)
    position = self.nominal_position + offset
    self.model.cam_pos[self.camera_id] = position
    return position


def add_depth_obstruction(
  spec: mujoco.MjSpec,
  depth_cfg: CameraSensorCfg,
  *,
  distance: float = 0.18,
  half_size: tuple[float, float, float] = (0.015, 0.035, 0.035),
  lateral_offset: float = 0.05,
  vertical_offset: float = 0.06,
) -> None:
  """Rigidly attach a small occluder in front of the depth camera lens.

  The box is a child of the camera's own parent body at the camera's own
  pose, offset along the camera's local -Z (forward, MuJoCo convention), so
  it stays in the same relative spot in the depth image regardless of how
  the robot moves -- like mud on the lens or a hand held in front of the
  camera, rather than a one-off obstacle the robot walks past. It sits in
  geom group 0, which is inside every depth camera's ``enabled_geom_groups``
  in this task family, so it always shows up in the depth render; contype
  and conaffinity are 0 so it never touches physics.

  ``lateral_offset``/``vertical_offset`` (camera-local +X/+Y, i.e. right/up)
  push the box toward the upper-right corner of the frame by default. This
  camera pitches steeply down at the ground in front of the robot, so the
  ball sits in the lower-center of the depth image; a dead-center occluder
  (offsets both 0) would sit right on top of the ball, which tests "what if
  the policy loses the ball" rather than "what if part of the frame the
  policy doesn't need is permanently blocked" -- push it to a corner to ask
  the latter question.

  Sized to cover a meaningful fraction of the frame (default: roughly a
  third of the vertical FOV at this distance) without blacking the whole
  thing out -- the interesting question is whether the policy tolerates
  *partial* permanent occlusion, not whether an all-invalid depth image
  (already exercised by the training-time dropout/artifact randomization)
  breaks it.
  """
  parent_body = depth_cfg.parent_body
  if parent_body is None:
    raise ValueError("Depth camera has no parent_body to attach the obstruction to.")
  parent = spec.body(parent_body)
  if parent is None:
    raise ValueError(f"MJLab scene is missing body {parent_body!r}.")

  forward_local = np.zeros(3)
  right_local = np.zeros(3)
  up_local = np.zeros(3)
  quat = np.array(depth_cfg.quat)
  mujoco.mju_rotVecQuat(forward_local, np.array([0.0, 0.0, -1.0]), quat)
  mujoco.mju_rotVecQuat(right_local, np.array([1.0, 0.0, 0.0]), quat)
  mujoco.mju_rotVecQuat(up_local, np.array([0.0, 1.0, 0.0]), quat)
  occluder_pos = (
    np.array(depth_cfg.pos)
    + distance * forward_local
    + lateral_offset * right_local
    + vertical_offset * up_local
  )
  parent.add_geom(
    name="depth_obstruction",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=occluder_pos,
    quat=depth_cfg.quat,
    size=half_size,
    group=0,
    contype=0,
    conaffinity=0,
    rgba=(0.8, 0.1, 0.1, 1.0),
  )


def build_model(
  *, task_id: str = TASK_ID, obstruct_depth: bool = False
) -> tuple[mujoco.MjModel, float, int, CameraSensorCfg]:
  """Compile the football play scene with its depth camera already attached.

  The camera is not added by hand (unlike the D435 RGB camera in
  ``g1_football.py``): it is part of ``env_cfg.scene.sensors`` for this task,
  so ``Scene`` inserts it into the MjSpec at the exact pose/fovy/resolution
  training uses, and geom-group filtering can be read back from the same cfg.
  """
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  depth_cfg = next(
    (
      sensor
      for sensor in env_cfg.scene.sensors or ()
      if sensor.name == DEPTH_SENSOR_NAME
    ),
    None,
  )
  if not isinstance(depth_cfg, CameraSensorCfg):
    raise ValueError(f"Task {task_id!r} has no {DEPTH_SENSOR_NAME!r} camera sensor.")
  scene = Scene(env_cfg.scene, device="cpu")
  if obstruct_depth:
    add_depth_obstruction(scene.spec, depth_cfg)
  model = scene.compile()
  env_cfg.sim.mujoco.apply(model)
  return model, model.opt.timestep, env_cfg.decimation, depth_cfg


def find_latest_policy(
  log_root: Path = Path("logs/rsl_rl"), *, task_id: str = TASK_ID
) -> Path:
  """Find the newest ONNX export compatible with the (proprio, depth) contract."""
  experiment_dir = log_root / load_rl_cfg(task_id).experiment_name
  policies = sorted(
    (path for path in experiment_dir.rglob("*.onnx") if path.is_file()),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
  )
  for path in policies:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    try:
      DepthPolicyMetadata.from_session(session)
    except ValueError:
      continue
    return path
  if not policies:
    raise FileNotFoundError(
      f"No ONNX policy found below {experiment_dir}. Train and let "
      "DepthTeacherDistillationRunner.save() export one, or pass --policy "
      "explicitly."
    )
  raise FileNotFoundError(
    f"No ONNX export below {experiment_dir} matches the (proprio, depth) "
    "Student contract."
  )


def _proprio_terms(
  data: mujoco.MjData,
  bindings: ModelBindings,
  default_joint_pos: FloatArray,
  command: FloatArray,
  phase: FloatArray,
  last_action: FloatArray,
) -> dict[str, FloatArray]:
  root_quat = data.xquat[bindings.root_body_id]
  gravity = quat_apply_inverse(root_quat, (0.0, 0.0, -1.0))
  joint_pos = data.qpos[bindings.joint_qpos_adr].astype(np.float32)
  joint_vel = data.qvel[bindings.joint_dof_adr].astype(np.float32)
  imu_slice = slice(bindings.imu_sensor_adr, bindings.imu_sensor_adr + 3)
  return {
    "base_ang_vel": data.sensordata[imu_slice].astype(np.float32),
    "projected_gravity": gravity,
    "command": command.astype(np.float32, copy=True),
    "phase": phase,
    "joint_pos": joint_pos - default_joint_pos,
    "joint_vel": joint_vel,
    "actions": np.clip(last_action, -10.0, 10.0).astype(np.float32),
  }


def _render_depth_frame(
  renderer: mujoco.Renderer,
  data: mujoco.MjData,
  *,
  camera_name: str,
  scene_option: mujoco.MjvOption,
) -> FloatArray:
  renderer.enable_depth_rendering()
  renderer.update_scene(data, camera=camera_name, scene_option=scene_option)
  depth = np.asarray(renderer.render(), dtype=np.float32).copy()
  renderer.disable_depth_rendering()
  return depth


def _reset(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  bindings: ModelBindings,
  metadata: DepthPolicyMetadata,
  renderer: mujoco.Renderer,
  scene_option: mujoco.MjvOption,
  user_command: FloatArray,
  proprio_assembler: ProprioAssembler,
  depth_history: _DepthHistoryBuffer,
  step_dt: float,
  depth_latency: DepthLatencyQueue | None,
  action_processor: TrainingActionProcessor,
  camera_position_randomizer: CameraPositionResetRandomizer | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
  if camera_position_randomizer is not None:
    position = camera_position_randomizer.reset()
    print(
      "Depth camera reset position: "
      f"[{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]"
    )
  mujoco.mj_resetDataKeyframe(model, data, bindings.init_key_id)
  data.ctrl[bindings.actuator_ids] = action_processor.reset()
  mujoco.mj_forward(model, data)
  action = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)
  terms = _proprio_terms(
    data,
    bindings,
    metadata.default_joint_pos,
    user_command,
    _phase(0, step_dt, user_command),
    action,
  )
  proprio_obs = proprio_assembler.reset(terms)
  depth_raw = _render_depth_frame(
    renderer, data, camera_name=DEPTH_SENSOR_NAME, scene_option=scene_option
  )
  depth_frame = normalize_and_downsample_depth(
    depth_raw,
    min_depth=DEPTH_MIN_METERS,
    max_depth=DEPTH_MAX_METERS,
    out_height=metadata.depth_height,
    out_width=metadata.depth_width,
  )
  if depth_latency is not None:
    depth_frame = depth_latency.reset(depth_frame)
  depth_history.reset(depth_frame)
  return action, proprio_obs, depth_history.stacked()


@dataclass(frozen=True)
class Sim2SimCfg:
  """Command-line configuration for the depth-image native MuJoCo deployment."""

  policy: Path | None = None
  task_id: str = TASK_ID
  """Registered play environment used to construct the robot and depth camera."""
  log_root: Path = Path("logs/rsl_rl")
  duration: float = 120.0
  headless: bool = False
  command_x: float = 0.5
  command_y: float = 0.0
  command_yaw: float = 0.0
  command_stop_time: float | None = None
  camera_distance: float = 3.0
  camera_azimuth: float = 90.0
  camera_elevation: float = -5.0
  camera_view: Literal["depth", "tracking"] = "tracking"
  show_depth_window: bool = True
  depth_window_rate: float = 15.0
  obstruct_depth: bool = False
  """Attach a small occluder in front of the depth camera lens (see
  ``add_depth_obstruction``) to probe robustness to partial, persistent
  depth occlusion."""
  depth_latency_steps: int = 0
  """Deliver the depth frame from this many policy steps ago instead of the
  one just rendered, modeling the async camera/inference pipeline real
  deployment has and sim2sim's synchronous render does not (see
  ``DepthLatencyQueue``). One step is 1/policy_rate, typically 20ms."""
  depth_jitter_steps: int = 0
  """Extra random delay (0 to this many steps, resampled every step) added
  on top of ``depth_latency_steps``, for irregular rather than constant
  staleness."""
  depth_latency_seed: int = 0
  """Seed for the jitter RNG, so a given run is reproducible."""
  camera_position_jitter_meters: float = 0.0
  """Per-axis translation sampled around calibrated pose on every reset."""
  camera_position_seed: int = 42
  """Seed for reset-time camera-position randomization."""

  def __post_init__(self) -> None:
    command = np.asarray(
      [self.command_x, self.command_y, self.command_yaw], dtype=np.float32
    )
    if not np.all(np.isfinite(command)):
      raise ValueError("Velocity command must contain only finite values.")
    if np.any(command < DEPLOYED_COMMAND_MIN) or np.any(command > DEPLOYED_COMMAND_MAX):
      raise ValueError(
        "Velocity command is outside the deployment range: "
        "vx=[-0.25, 1.0], vy=[-0.25, 0.25], yaw=[-1.0, 1.0]."
      )
    if self.depth_latency_steps < 0:
      raise ValueError(
        f"depth_latency_steps must be non-negative, got {self.depth_latency_steps}"
      )
    if self.depth_jitter_steps < 0:
      raise ValueError(
        f"depth_jitter_steps must be non-negative, got {self.depth_jitter_steps}"
      )
    if self.camera_position_jitter_meters < 0.0:
      raise ValueError(
        "camera_position_jitter_meters must be non-negative, got "
        f"{self.camera_position_jitter_meters}"
      )


def run(cfg: Sim2SimCfg) -> None:
  """Load a depth-image policy and execute it in native MuJoCo."""
  policy_path = (
    cfg.policy or find_latest_policy(cfg.log_root, task_id=cfg.task_id)
  ).resolve()
  if not policy_path.is_file():
    raise FileNotFoundError(f"ONNX policy does not exist: {policy_path}")
  session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
  metadata = DepthPolicyMetadata.from_session(session)

  model, timestep, decimation, depth_cfg = build_model(
    task_id=cfg.task_id, obstruct_depth=cfg.obstruct_depth
  )
  data = mujoco.MjData(model)
  bindings = ModelBindings.from_model(model, metadata.joint_names)
  action_processor = TrainingActionProcessor(metadata)
  camera_position_randomizer = (
    CameraPositionResetRandomizer(
      model=model,
      camera_id=bindings.depth_camera_id,
      nominal_position=model.cam_pos[bindings.depth_camera_id]
      .astype(np.float32)
      .copy(),
      jitter_meters=cfg.camera_position_jitter_meters,
      rng=np.random.default_rng(cfg.camera_position_seed),
    )
    if cfg.camera_position_jitter_meters > 0.0
    else None
  )

  renderer = mujoco.Renderer(model, height=depth_cfg.height, width=depth_cfg.width)
  scene_option = mujoco.MjvOption()
  scene_option.geomgroup[:] = 0
  for group in depth_cfg.enabled_geom_groups:
    scene_option.geomgroup[group] = 1

  proprio_assembler = ProprioAssembler()
  depth_history = _DepthHistoryBuffer(metadata.depth_history_length)
  depth_latency = (
    DepthLatencyQueue(
      cfg.depth_latency_steps,
      cfg.depth_jitter_steps,
      np.random.default_rng(cfg.depth_latency_seed),
    )
    if cfg.depth_latency_steps > 0 or cfg.depth_jitter_steps > 0
    else None
  )
  command = np.asarray(
    [cfg.command_x, cfg.command_y, cfg.command_yaw], dtype=np.float32
  )
  keyboard = KeyboardController(
    command,
    command_min=DEPLOYED_COMMAND_MIN,
    command_max=DEPLOYED_COMMAND_MAX,
  )
  step_dt = timestep * decimation
  action, proprio_obs, depth_obs = _reset(
    model,
    data,
    bindings,
    metadata,
    renderer,
    scene_option,
    keyboard.command,
    proprio_assembler,
    depth_history,
    step_dt,
    depth_latency,
    action_processor,
    camera_position_randomizer,
  )
  policy_step = 0
  total_policy_steps = max(0, int(cfg.duration / step_dt))
  stop_command_applied = False

  viewer: Any = None
  depth_window: DepthWindow | None = None
  if not cfg.headless:
    from mujoco import viewer as mujoco_viewer

    viewer = mujoco_viewer.launch_passive(model, data, key_callback=keyboard)
    # Draw the depth camera's axis triad, frustum, and name label in the main
    # 3D view so the newly calibrated mount pose is visible, not just implied.
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_CAMERA
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_CAMERA
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
    if cfg.camera_view == "depth":
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      viewer.cam.fixedcamid = bindings.depth_camera_id
    else:
      configure_tracking_camera(
        viewer.cam,
        bindings.root_body_id,
        distance=cfg.camera_distance,
        azimuth=cfg.camera_azimuth,
        elevation=cfg.camera_elevation,
      )
    print("Controls: 8/2 forward, 4/6 lateral, 7/9 yaw, 5 stop, R reset")
    if cfg.show_depth_window:
      try:
        depth_window = DepthWindow(
          metadata.depth_height,
          metadata.depth_width,
          DEPTH_MIN_METERS,
          DEPTH_MAX_METERS,
          cfg.depth_window_rate,
        )
      except Exception as exc:  # pragma: no cover - GUI backend dependent.
        print(f"Depth window unavailable: {exc}")

  print(f"Policy: {policy_path}")
  print(f"Environment task: {cfg.task_id}")
  print(
    f"Native MuJoCo: dt={timestep:.3f}s, decimation={decimation}, "
    f"policy_rate={1.0 / step_dt:.1f}Hz"
  )
  print(
    "Depth observation: "
    f"history={metadata.depth_history_length}, "
    f"size={metadata.depth_height}x{metadata.depth_width}, "
    f"range=[{DEPTH_MIN_METERS:.2f}, {DEPTH_MAX_METERS:.2f}] m, "
    f"raw={depth_cfg.height}x{depth_cfg.width}, "
    f"geom_groups={tuple(depth_cfg.enabled_geom_groups)}"
  )
  print("Action processing: training parity (no action clamps)")
  if cfg.obstruct_depth:
    print("Depth obstruction: ON (occluder fixed in front of the lens)")
  if depth_latency is not None:
    print(
      "Depth latency: ON "
      f"(base={cfg.depth_latency_steps} steps, "
      f"jitter=0-{cfg.depth_jitter_steps} steps, "
      f"~{step_dt * 1000:.0f}ms/step)"
    )
  if camera_position_randomizer is not None:
    print(
      "Depth camera reset randomization: ON "
      f"(per-axis +/-{cfg.camera_position_jitter_meters * 1000:.1f}mm)"
    )
  try:
    while policy_step < total_policy_steps:
      if viewer is not None and not viewer.is_running():
        break
      started = time.perf_counter()
      if keyboard.reset_requested:
        keyboard.reset_requested = False
        policy_step = 0
        stop_command_applied = False
        action, proprio_obs, depth_obs = _reset(
          model,
          data,
          bindings,
          metadata,
          renderer,
          scene_option,
          keyboard.command,
          proprio_assembler,
          depth_history,
          step_dt,
          depth_latency,
          action_processor,
          camera_position_randomizer,
        )

      if (
        cfg.command_stop_time is not None
        and not stop_command_applied
        and data.time >= cfg.command_stop_time
      ):
        keyboard.command[:] = 0.0
        stop_command_applied = True
        print(
          f"Command stop applied at t={data.time:.3f}s "
          f"(configured {cfg.command_stop_time:.3f}s)."
        )

      policy_output = session.run(
        ["actions"],
        {
          "proprio": proprio_obs.reshape(1, -1),
          "depth": depth_obs.reshape(
            1,
            metadata.depth_history_length,
            metadata.depth_height,
            metadata.depth_width,
          ),
        },
      )[0]
      action = np.asarray(policy_output, dtype=np.float32)[0]
      target = action_processor.process(action)
      data.ctrl[bindings.actuator_ids] = target
      for _ in range(decimation):
        mujoco.mj_step(model, data)
        if viewer is not None:
          viewer.sync()

      policy_step += 1
      terms = _proprio_terms(
        data,
        bindings,
        metadata.default_joint_pos,
        keyboard.command,
        _phase(policy_step, step_dt, keyboard.command),
        action,
      )
      proprio_obs = proprio_assembler.append(terms)
      depth_raw = _render_depth_frame(
        renderer, data, camera_name=DEPTH_SENSOR_NAME, scene_option=scene_option
      )
      depth_frame = normalize_and_downsample_depth(
        depth_raw,
        min_depth=DEPTH_MIN_METERS,
        max_depth=DEPTH_MAX_METERS,
        out_height=metadata.depth_height,
        out_width=metadata.depth_width,
      )
      delivered_frame = (
        depth_latency.push(depth_frame) if depth_latency is not None else depth_frame
      )
      depth_history.append(delivered_frame)
      depth_obs = depth_history.stacked()
      if depth_window is not None:
        depth_window.update(delivered_frame, float(data.time))
      if viewer is not None:
        remaining = step_dt - (time.perf_counter() - started)
        if remaining > 0.0:
          time.sleep(remaining)
  finally:
    renderer.close()
    if depth_window is not None:
      depth_window.close()
    if viewer is not None:
      viewer.close()


def main() -> None:
  """Parse command-line arguments and run depth-image football sim-to-sim."""
  run(tyro.cli(Sim2SimCfg, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
