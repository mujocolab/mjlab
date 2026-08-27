"""Run the MJLab G1 football policy in native CPU MuJoCo."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
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
from mjlab.scripts.sim2sim.d435_ball_observer import (
  D435_CAMERA_NAME,
  D435BallObserver,
  D435Config,
  add_d435_camera,
  add_football_visual_material,
  make_ball_observer,
  world_to_yaw,
)
from mjlab.scripts.sim2sim.detection_window import DetectionWindow
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity_football.config.g1 import TEACHER_BASELINE_TASK_ID

TASK_ID = TEACHER_BASELINE_TASK_ID
FRAME_STACK = 5
TEMPORAL_HISTORY_LENGTH = 10
BALL_VISIBILITY_X_RANGE = (0.05, 1.00)
BALL_VISIBILITY_Y_RANGE = (-0.70, 0.70)
PHASE_PERIOD = 0.6
TRAINED_COMMAND_MIN = np.asarray([-0.5, -0.5, -1.0], dtype=np.float32)
TRAINED_COMMAND_MAX = np.asarray([2.0, 0.5, 1.0], dtype=np.float32)

TERM_DIMS: dict[str, int] = {
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "command": 3,
  "phase": 2,
  "joint_pos": 29,
  "joint_vel": 29,
  "actions": 29,
  "ball_pos_b": 2,
  "ball_to_feet_vectors_b": 4,
}
TEMPORAL_TERM_DIMS = {
  **TERM_DIMS,
  "ball_visible_mask": 1,
}
EXPECTED_OBSERVATION_NAMES = tuple(TERM_DIMS)
EXPECTED_OBS_DIM = FRAME_STACK * sum(TERM_DIMS.values())
PROPRIOCEPTIVE_OBSERVATION_NAMES = EXPECTED_OBSERVATION_NAMES[:-2]
PROPRIOCEPTIVE_OBS_DIM = FRAME_STACK * sum(
  TERM_DIMS[name] for name in PROPRIOCEPTIVE_OBSERVATION_NAMES
)
TEMPORAL_OBSERVATION_NAMES = tuple(TEMPORAL_TERM_DIMS)
TEMPORAL_OBS_DIM = sum(TEMPORAL_TERM_DIMS.values())
ISAACLAB_ALIGNED_OBS_DIM = FRAME_STACK * TEMPORAL_OBS_DIM
B1_HISTORY_TERM_DIMS = {
  "ball_pos_b": 2,
  "ball_to_feet_vectors_b": 4,
  "ball_visible_mask": 1,
}
B1_HISTORY_OBSERVATION_NAMES = tuple(B1_HISTORY_TERM_DIMS)
B1_HISTORY_OBS_DIM = sum(B1_HISTORY_TERM_DIMS.values())
EXPECTED_ACTION_DIM = TERM_DIMS["actions"]

FloatArray = npt.NDArray[np.float32]


def _parse_csv(metadata: dict[str, str], key: str) -> tuple[str, ...]:
  try:
    value = metadata[key]
  except KeyError as exc:
    raise ValueError(f"ONNX metadata is missing required field {key!r}.") from exc
  return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_float_csv(metadata: dict[str, str], key: str) -> FloatArray:
  values = _parse_csv(metadata, key)
  try:
    return np.asarray(values, dtype=np.float32)
  except ValueError as exc:
    raise ValueError(f"ONNX metadata field {key!r} is not numeric.") from exc


@dataclass(frozen=True)
class PolicyMetadata:
  """Deployment parameters embedded in an exported MJLab ONNX policy."""

  joint_names: tuple[str, ...]
  default_joint_pos: FloatArray
  action_scale: FloatArray
  observation_names: tuple[str, ...]
  observation_history: tuple[int, ...]
  observation_dim: int
  temporal_history_length: int | None = None
  temporal_history_dim: int | None = None

  @property
  def is_temporal(self) -> bool:
    return self.temporal_history_length is not None

  @classmethod
  def from_session(cls, session: Any) -> PolicyMetadata:
    """Parse and validate the deployment contract from an ONNX session."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_names = tuple(item.name for item in inputs)
    legacy = input_names == ("obs",)
    temporal = input_names == ("obs", "obs_history")
    if not legacy and not temporal:
      raise ValueError(
        f"Policy inputs must be ('obs',) or ('obs', 'obs_history'), got {input_names}."
      )
    if len(outputs) != 1 or outputs[0].name != "actions":
      raise ValueError("Policy must expose exactly one ONNX output named 'actions'.")

    input_dim = inputs[0].shape[-1]
    output_dim = outputs[0].shape[-1]
    if not temporal and input_dim not in {
      EXPECTED_OBS_DIM,
      ISAACLAB_ALIGNED_OBS_DIM,
    }:
      raise ValueError(
        f"Policy expects {input_dim} observations; this task requires either "
        f"{EXPECTED_OBS_DIM} (legacy) or {ISAACLAB_ALIGNED_OBS_DIM} "
        "(IsaacLab-aligned visibility mask)."
      )
    temporal_history_length = None
    temporal_history_dim = None
    if temporal:
      history_shape = tuple(inputs[1].shape)
      if len(history_shape) != 3 or history_shape[-1] not in {
        TEMPORAL_OBS_DIM,
        B1_HISTORY_OBS_DIM,
      }:
        raise ValueError(
          "Temporal policy obs_history must have shape "
          "(batch, history_length, history_dim), where history_dim is "
          f"{TEMPORAL_OBS_DIM} (legacy full history) or "
          f"{B1_HISTORY_OBS_DIM} (B1 ball history), "
          f"got {history_shape}."
        )
      history_length = history_shape[-2]
      if not isinstance(history_length, int) or history_length <= 0:
        raise ValueError(
          "Temporal policy obs_history must expose a fixed positive history "
          f"length, got {history_shape}."
        )
      temporal_history_length = history_length
      temporal_history_dim = history_shape[-1]
      supported_contract = (input_dim, temporal_history_dim) in {
        (TEMPORAL_OBS_DIM, TEMPORAL_OBS_DIM),
        (TEMPORAL_OBS_DIM, B1_HISTORY_OBS_DIM),
        (PROPRIOCEPTIVE_OBS_DIM, B1_HISTORY_OBS_DIM),
      }
      if not supported_contract:
        raise ValueError(
          "Unsupported temporal policy input contract: "
          f"obs={input_dim}, obs_history={temporal_history_dim}."
        )
    if output_dim != EXPECTED_ACTION_DIM:
      raise ValueError(
        f"Policy produces {output_dim} actions; this task requires "
        f"{EXPECTED_ACTION_DIM}."
      )

    metadata = session.get_modelmeta().custom_metadata_map
    joint_names = _parse_csv(metadata, "joint_names")
    default_joint_pos = _parse_float_csv(metadata, "default_joint_pos")
    action_scale = _parse_float_csv(metadata, "action_scale")
    observation_names = _parse_csv(metadata, "observation_names")
    history_raw = _parse_float_csv(metadata, "observation_terms_history_length")
    observation_history = tuple(int(value) for value in history_raw)

    if len(joint_names) != EXPECTED_ACTION_DIM:
      raise ValueError(
        f"Policy metadata contains {len(joint_names)} joints; expected "
        f"{EXPECTED_ACTION_DIM}."
      )
    if default_joint_pos.shape != (EXPECTED_ACTION_DIM,):
      raise ValueError("ONNX default_joint_pos must contain 29 values.")
    if action_scale.shape != (EXPECTED_ACTION_DIM,):
      raise ValueError("ONNX action_scale must contain 29 values.")
    expected_names = EXPECTED_OBSERVATION_NAMES
    if not temporal and input_dim == ISAACLAB_ALIGNED_OBS_DIM:
      expected_names = TEMPORAL_OBSERVATION_NAMES
    elif temporal and input_dim == TEMPORAL_OBS_DIM:
      expected_names = TEMPORAL_OBSERVATION_NAMES
    elif temporal and input_dim == PROPRIOCEPTIVE_OBS_DIM:
      expected_names = PROPRIOCEPTIVE_OBSERVATION_NAMES
    if observation_names != expected_names:
      raise ValueError(
        "ONNX observation order does not match the MJLab football task: "
        f"{observation_names}."
      )
    expected_history = (
      (0,) * len(observation_names)
      if temporal and input_dim == TEMPORAL_OBS_DIM
      else (FRAME_STACK,) * len(observation_names)
    )
    if observation_history != expected_history:
      raise ValueError(
        "ONNX observation history metadata does not match its input contract."
      )

    return cls(
      joint_names=joint_names,
      default_joint_pos=default_joint_pos,
      action_scale=action_scale,
      observation_names=observation_names,
      observation_history=observation_history,
      observation_dim=input_dim,
      temporal_history_length=temporal_history_length,
      temporal_history_dim=temporal_history_dim,
    )


def align_legacy_metadata_to_task(
  metadata: PolicyMetadata,
  task_id: str,
) -> PolicyMetadata:
  """Repair exports whose names/defaults used natural rather than action order."""
  env_cfg = load_env_cfg(task_id, play=True)
  action_cfg = env_cfg.actions["joint_pos"]
  configured_names = tuple(action_cfg.actuator_names)
  if len(configured_names) != EXPECTED_ACTION_DIM or len(set(configured_names)) != len(
    configured_names
  ):
    return metadata
  if configured_names == metadata.joint_names:
    return metadata
  if set(configured_names) != set(metadata.joint_names):
    raise ValueError(
      "Task action joints and ONNX metadata joints do not describe the same set: "
      f"task={configured_names}, metadata={metadata.joint_names}."
    )
  natural_default = dict(
    zip(metadata.joint_names, metadata.default_joint_pos, strict=True)
  )
  corrected_default = np.asarray(
    [natural_default[name] for name in configured_names], dtype=np.float32
  )
  print(
    "[WARN] Correcting legacy ONNX joint metadata from natural MJCF order "
    "to the task's configured action order."
  )
  return replace(
    metadata,
    joint_names=configured_names,
    default_joint_pos=corrected_default,
  )


class _HistoryBuffer:
  """Chronological observation history stored oldest to newest."""

  def __init__(self, length: int) -> None:
    self._length = length
    self._values: list[FloatArray] = []

  def reset(self, value: FloatArray) -> None:
    self._values = [value.copy() for _ in range(self._length)]

  def append(self, value: FloatArray) -> None:
    if not self._values:
      self.reset(value)
      return
    self._values.append(value.copy())
    self._values.pop(0)

  def flatten(self, length: int | None = None) -> FloatArray:
    if not self._values:
      raise RuntimeError("Observation history has not been initialized.")
    values = self._values if length is None else self._values[-length:]
    return np.concatenate(values).astype(np.float32, copy=False)


class ObservationAssembler:
  """Reproduce deployment-time per-term history semantics for the actor."""

  def __init__(self, metadata: PolicyMetadata | None = None) -> None:
    self._temporal = metadata is not None and metadata.is_temporal
    uses_visibility_mask = metadata is not None and (
      "ball_visible_mask" in metadata.observation_names
    )
    self._term_dims = (
      TEMPORAL_TERM_DIMS if self._temporal or uses_visibility_mask else TERM_DIMS
    )
    self._observation_names = (
      metadata.observation_names if metadata is not None else EXPECTED_OBSERVATION_NAMES
    )
    self._all_names = tuple(self._term_dims)
    history_length = (
      metadata.temporal_history_length
      if self._temporal and metadata is not None
      else FRAME_STACK
    )
    assert history_length is not None
    self._history_length = history_length
    buffer_length = max(FRAME_STACK, history_length)
    self._history_names = self._observation_names
    self._history_dim = TEMPORAL_OBS_DIM
    if self._temporal and metadata is not None:
      assert metadata.temporal_history_dim is not None
      self._history_dim = metadata.temporal_history_dim
      if self._history_dim == B1_HISTORY_OBS_DIM:
        self._history_names = B1_HISTORY_OBSERVATION_NAMES
    self._current_dim = (
      metadata.observation_dim if metadata is not None else EXPECTED_OBS_DIM
    )
    self._history = {name: _HistoryBuffer(buffer_length) for name in self._all_names}

  def _validate_terms(self, terms: dict[str, FloatArray]) -> None:
    if tuple(terms) != self._all_names:
      raise ValueError(f"Observation terms must be ordered as {self._all_names}.")
    for name, expected_dim in self._term_dims.items():
      if terms[name].shape != (expected_dim,):
        raise ValueError(
          f"Observation term {name!r} has shape {terms[name].shape}; "
          f"expected ({expected_dim},)."
        )

  def reset(self, terms: dict[str, FloatArray]) -> FloatArray:
    """Backfill every history buffer using the reset observation."""
    self._validate_terms(terms)
    for name, value in terms.items():
      self._history[name].reset(value)
    return self.observation()

  def append(self, terms: dict[str, FloatArray]) -> FloatArray:
    """Append one policy-step observation and return the flattened history."""
    self._validate_terms(terms)
    for name, value in terms.items():
      self._history[name].append(value)
    return self.observation()

  def observation(self) -> FloatArray:
    """Return the current frame or legacy term-major flattened history."""
    if self._temporal and self._current_dim == TEMPORAL_OBS_DIM:
      obs = np.concatenate(
        [self._history[name]._values[-1] for name in self._observation_names]
      ).astype(np.float32, copy=False)
      expected_dim = TEMPORAL_OBS_DIM
    else:
      obs = np.concatenate(
        [self._history[name].flatten(FRAME_STACK) for name in self._observation_names]
      ).astype(np.float32, copy=False)
      expected_dim = self._current_dim
    if obs.shape != (expected_dim,):
      raise RuntimeError(f"Assembled invalid observation shape {obs.shape}.")
    if not np.all(np.isfinite(obs)):
      raise RuntimeError("Sim-to-sim observation contains NaN or Inf values.")
    return obs

  def policy_inputs(self, obs: FloatArray) -> dict[str, Any]:
    """Build the ONNX feed dictionary for either supported policy contract."""
    inputs: dict[str, Any] = {"obs": obs.reshape(1, -1)}
    if self._temporal:
      history = np.stack(
        [
          np.concatenate(
            [self._history[name]._values[index] for name in self._observation_names]
            if self._history_dim == TEMPORAL_OBS_DIM
            else [self._history[name]._values[index] for name in self._history_names]
          )
          for index in range(-self._history_length, 0)
        ]
      ).astype(np.float32, copy=False)
      inputs["obs_history"] = history.reshape(
        1, self._history_length, self._history_dim
      )
    return inputs


def quat_apply_inverse(quat_wxyz: npt.ArrayLike, vector: npt.ArrayLike) -> FloatArray:
  """Rotate a vector from world coordinates into a quaternion's local frame."""
  quat = np.asarray(quat_wxyz, dtype=np.float64)
  vec = np.asarray(vector, dtype=np.float64)
  quat = quat / np.linalg.norm(quat)
  xyz = -quat[1:]
  t = 2.0 * np.cross(xyz, vec)
  rotated = vec + quat[0] * t + np.cross(xyz, t)
  return rotated.astype(np.float32)


def compute_football_observation(
  root_pos_w: npt.ArrayLike,
  root_quat_w: npt.ArrayLike,
  ball_pos_w: npt.ArrayLike,
  feet_pos_w: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
  """Compute yaw-aligned XY observations using the foot-to-football convention."""
  root_pos = np.asarray(root_pos_w, dtype=np.float64)
  ball_pos = np.asarray(ball_pos_w, dtype=np.float64)
  feet_pos = np.asarray(feet_pos_w, dtype=np.float64)
  ball_pos_yaw = world_to_yaw(ball_pos - root_pos, root_quat_w)
  feet_pos_yaw = np.stack(
    [world_to_yaw(foot_pos - root_pos, root_quat_w) for foot_pos in feet_pos]
  )
  feet_to_ball_yaw = ball_pos_yaw[None, :2] - feet_pos_yaw[:, :2]
  return ball_pos_yaw[:2], feet_to_ball_yaw.astype(np.float32).reshape(-1)


@dataclass(frozen=True)
class ModelBindings:
  """Name-based indices connecting policy order to a compiled MuJoCo model."""

  joint_qpos_adr: npt.NDArray[np.int32]
  joint_dof_adr: npt.NDArray[np.int32]
  actuator_ids: npt.NDArray[np.int32]
  root_body_id: int
  ball_body_id: int
  ball_dof_adr: int
  foot_body_ids: npt.NDArray[np.int32]
  imu_sensor_adr: int
  d435_camera_id: int
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
    foot_body_ids = np.asarray(
      [
        require_id(mujoco.mjtObj.mjOBJ_BODY, "robot/left_ankle_roll_link"),
        require_id(mujoco.mjtObj.mjOBJ_BODY, "robot/right_ankle_roll_link"),
      ],
      dtype=np.int32,
    )
    imu_sensor_id = require_id(mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")
    if model.sensor_dim[imu_sensor_id] != 3:
      raise ValueError("robot/imu_ang_vel must be a three-dimensional sensor.")
    ball_joint_id = require_id(mujoco.mjtObj.mjOBJ_JOINT, "ball/ball_freejoint")
    if model.jnt_type[ball_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
      raise ValueError("ball/ball_freejoint must be a MuJoCo free joint.")
    return cls(
      joint_qpos_adr=model.jnt_qposadr[joint_ids].astype(np.int32),
      joint_dof_adr=model.jnt_dofadr[joint_ids].astype(np.int32),
      actuator_ids=actuator_ids,
      root_body_id=require_id(mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis"),
      ball_body_id=require_id(mujoco.mjtObj.mjOBJ_BODY, "ball/ball"),
      ball_dof_adr=int(model.jnt_dofadr[ball_joint_id]),
      foot_body_ids=foot_body_ids,
      imu_sensor_adr=int(model.sensor_adr[imu_sensor_id]),
      d435_camera_id=require_id(mujoco.mjtObj.mjOBJ_CAMERA, D435_CAMERA_NAME),
      init_key_id=require_id(mujoco.mjtObj.mjOBJ_KEY, "init_state"),
    )


def build_model(
  d435_cfg: D435Config | None = None,
  task_id: str = TASK_ID,
) -> tuple[mujoco.MjModel, float, int]:
  """Compile the nominal football play scene and apply its simulation options."""
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  scene = Scene(env_cfg.scene, device="cpu")
  add_d435_camera(scene.spec, d435_cfg or D435Config())
  add_football_visual_material(scene.spec)
  model = scene.compile()
  env_cfg.sim.mujoco.apply(model)
  return model, model.opt.timestep, env_cfg.decimation


def find_latest_policy(log_root: Path = Path("logs/rsl_rl")) -> Path:
  """Find the newest ONNX export compatible with a supported flat contract."""
  experiment_name = load_rl_cfg(TASK_ID).experiment_name
  experiment_dir = log_root / experiment_name
  policies = sorted(
    (path for path in experiment_dir.rglob("*.onnx") if path.is_file()),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
  )
  for path in policies:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] in {
      EXPECTED_OBS_DIM,
      ISAACLAB_ALIGNED_OBS_DIM,
    }:
      return path
  if not policies:
    raise FileNotFoundError(
      f"No ONNX policy found below {experiment_dir}. Run play once to export it, "
      "or pass --policy explicitly."
    )
  raise FileNotFoundError(
    f"No supported {EXPECTED_OBS_DIM}- or {ISAACLAB_ALIGNED_OBS_DIM}-input "
    f"ONNX policy found below {experiment_dir}. "
    "The existing 535-input exports use the previous football observation contract."
  )


class KeyboardController:
  """Mutable velocity command controlled by the native MuJoCo viewer."""

  def __init__(
    self,
    command: FloatArray,
    command_min: FloatArray = TRAINED_COMMAND_MIN,
    command_max: FloatArray = TRAINED_COMMAND_MAX,
  ) -> None:
    self.command = command.copy()
    self._command_min = command_min.copy()
    self._command_max = command_max.copy()
    self.reset_requested = False

  def __call__(self, keycode: int) -> None:
    if 320 <= keycode <= 329:  # GLFW_KEY_KP_0 through GLFW_KEY_KP_9.
      key = str(keycode - 320)
    else:
      key = chr(keycode).upper() if 0 <= keycode < 256 else ""
    if key == "8":
      self.command[0] += 0.1
    elif key == "2":
      self.command[0] -= 0.1
    elif key == "4":
      self.command[1] += 0.1
    elif key == "6":
      self.command[1] -= 0.1
    elif key == "7":
      self.command[2] += 0.1
    elif key == "9":
      self.command[2] -= 0.1
    elif key == "5":
      self.command[:] = 0.0
    elif key == "R":
      self.reset_requested = True
    self.command[:] = np.clip(
      self.command,
      self._command_min,
      self._command_max,
    )


def configure_tracking_camera(
  camera: mujoco.MjvCamera,
  body_id: int,
  *,
  distance: float,
  azimuth: float,
  elevation: float,
) -> None:
  """Configure a stable view that follows a body without inheriting its pose."""
  camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  camera.trackbodyid = body_id
  camera.distance = distance
  camera.azimuth = azimuth
  camera.elevation = elevation


def _phase(step: int, step_dt: float, command: FloatArray) -> FloatArray:
  if np.linalg.norm(command) < 0.1:
    return np.zeros(2, dtype=np.float32)
  value = (step * step_dt) % PHASE_PERIOD / PHASE_PERIOD
  return np.asarray(
    [np.sin(2.0 * np.pi * value), np.cos(2.0 * np.pi * value)],
    dtype=np.float32,
  )


def _observation_terms(
  data: mujoco.MjData,
  bindings: ModelBindings,
  metadata: PolicyMetadata,
  command: FloatArray,
  phase: FloatArray,
  last_action: FloatArray,
  football_observation: tuple[FloatArray, FloatArray],
) -> dict[str, FloatArray]:
  root_quat = data.xquat[bindings.root_body_id]
  ball_pos, feet_to_ball = football_observation
  gravity = quat_apply_inverse(root_quat, (0.0, 0.0, -1.0))
  joint_pos = data.qpos[bindings.joint_qpos_adr].astype(np.float32)
  joint_vel = data.qvel[bindings.joint_dof_adr].astype(np.float32)
  imu_slice = slice(bindings.imu_sensor_adr, bindings.imu_sensor_adr + 3)
  terms = {
    "base_ang_vel": data.sensordata[imu_slice].astype(np.float32),
    "projected_gravity": gravity,
    "command": command.astype(np.float32, copy=True),
    "phase": phase,
    "joint_pos": joint_pos - metadata.default_joint_pos,
    "joint_vel": joint_vel,
    "actions": last_action.astype(np.float32, copy=True),
    "ball_pos_b": ball_pos,
    "ball_to_feet_vectors_b": feet_to_ball,
  }
  if metadata.is_temporal or "ball_visible_mask" in metadata.observation_names:
    true_ball_pos, _ = compute_football_observation(
      data.xpos[bindings.root_body_id],
      root_quat,
      data.xpos[bindings.ball_body_id],
      data.xpos[bindings.foot_body_ids],
    )
    visible = float(
      np.any(np.abs(ball_pos) > 1e-6)
      and BALL_VISIBILITY_X_RANGE[0]
      <= float(true_ball_pos[0])
      <= BALL_VISIBILITY_X_RANGE[1]
      and BALL_VISIBILITY_Y_RANGE[0]
      <= float(true_ball_pos[1])
      <= BALL_VISIBILITY_Y_RANGE[1]
    )
    terms["ball_pos_b"] = ball_pos * visible
    terms["ball_to_feet_vectors_b"] = feet_to_ball * visible
    terms["ball_visible_mask"] = np.asarray([visible], dtype=np.float32)
  return terms


def _reset(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  bindings: ModelBindings,
  metadata: PolicyMetadata,
  user_command: FloatArray,
  assembler: ObservationAssembler,
  ball_observer: Any,
  step_dt: float,
) -> tuple[FloatArray, FloatArray]:
  mujoco.mj_resetDataKeyframe(model, data, bindings.init_key_id)
  data.ctrl[bindings.actuator_ids] = metadata.default_joint_pos
  mujoco.mj_forward(model, data)
  ball_observer.reset()
  action = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)
  football_observation = ball_observer.observe(data)
  terms = _observation_terms(
    data,
    bindings,
    metadata,
    user_command,
    _phase(0, step_dt, user_command),
    action,
    football_observation,
  )
  return action, assembler.reset(terms)


@dataclass(frozen=True)
class Sim2SimCfg:
  """Command-line configuration for native MuJoCo football deployment."""

  policy: Path | None = None
  task_id: str = TASK_ID
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
  camera_view: Literal["d435", "tracking"] = "d435"
  ball_observer: Literal["robocup", "d435", "mujoco"] = "robocup"
  yolo_model: Path | None = None
  yolo_confidence: float | None = None
  ball_hold_time: float = 0.5
  show_detection_window: bool = True
  detection_window_rate: float = 15.0

  def __post_init__(self) -> None:
    command = np.asarray(
      [self.command_x, self.command_y, self.command_yaw], dtype=np.float32
    )
    if not np.all(np.isfinite(command)):
      raise ValueError("Velocity command must contain only finite values.")
    if np.any(command < TRAINED_COMMAND_MIN) or np.any(command > TRAINED_COMMAND_MAX):
      raise ValueError(
        "Velocity command is outside the trained range: "
        "vx=[-0.5, 2.0], vy=[-0.5, 0.5], yaw=[-1.0, 1.0]."
      )


def run(cfg: Sim2SimCfg) -> None:
  """Load a policy and execute it in native MuJoCo."""
  policy_path = (cfg.policy or find_latest_policy(cfg.log_root)).resolve()
  if not policy_path.is_file():
    raise FileNotFoundError(f"ONNX policy does not exist: {policy_path}")
  session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
  metadata = PolicyMetadata.from_session(session)
  metadata = align_legacy_metadata_to_task(metadata, cfg.task_id)
  use_robocup_vision = cfg.ball_observer == "robocup"
  yolo_confidence = cfg.yolo_confidence
  if yolo_confidence is None:
    yolo_confidence = 0.2 if use_robocup_vision else 0.5
  d435_cfg = D435Config(
    confidence_threshold=yolo_confidence,
    iou_threshold=0.4 if use_robocup_vision else 0.5,
    max_hold_time=cfg.ball_hold_time,
    vision_mode="robocup" if use_robocup_vision else "deployment_rgbd",
  )
  model, timestep, decimation = build_model(d435_cfg, cfg.task_id)
  data = mujoco.MjData(model)
  bindings = ModelBindings.from_model(model, metadata.joint_names)
  ball_observer = make_ball_observer(
    cfg.ball_observer,
    model,
    root_body_id=bindings.root_body_id,
    ball_body_id=bindings.ball_body_id,
    foot_body_ids=bindings.foot_body_ids,
    yolo_model=cfg.yolo_model,
    cfg=d435_cfg,
  )
  d435_observer = ball_observer if isinstance(ball_observer, D435BallObserver) else None
  assembler = ObservationAssembler(metadata)
  command = np.asarray(
    [cfg.command_x, cfg.command_y, cfg.command_yaw], dtype=np.float32
  )
  keyboard = KeyboardController(command)
  step_dt = timestep * decimation
  action, obs = _reset(
    model,
    data,
    bindings,
    metadata,
    keyboard.command,
    assembler,
    ball_observer,
    step_dt,
  )
  policy_step = 0
  total_policy_steps = max(0, int(cfg.duration / step_dt))
  stop_command_applied = False

  viewer: Any = None
  detection_window: DetectionWindow | None = None
  if not cfg.headless:
    from mujoco import viewer as mujoco_viewer

    viewer = mujoco_viewer.launch_passive(model, data, key_callback=keyboard)
    if cfg.camera_view == "d435":
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      viewer.cam.fixedcamid = bindings.d435_camera_id
    else:
      configure_tracking_camera(
        viewer.cam,
        bindings.root_body_id,
        distance=cfg.camera_distance,
        azimuth=cfg.camera_azimuth,
        elevation=cfg.camera_elevation,
      )
    print("Controls: 8/2 forward, 4/6 lateral, 7/9 yaw, 5 stop, R reset")
    if cfg.show_detection_window and d435_observer is not None:
      try:
        detection_window = DetectionWindow(d435_cfg, cfg.detection_window_rate)
      except Exception as exc:  # pragma: no cover - GUI backend dependent.
        print(f"Detection window unavailable: {exc}")

  print(f"Policy: {policy_path}")
  print(f"Task model: {cfg.task_id}")
  print(
    f"Native MuJoCo: dt={timestep:.3f}s, decimation={decimation}, "
    f"policy_rate={1.0 / step_dt:.1f}Hz"
  )
  print(
    f"Football observation: source={cfg.ball_observer}, viewer_camera={cfg.camera_view}"
  )
  if cfg.ball_observer == "robocup":
    print(
      "RoboCup vision parity: top-left black padding, bbox-bottom ground "
      f"intersection, confidence={d435_cfg.confidence_threshold:.2f}, "
      f"NMS={d435_cfg.iou_threshold:.2f}, hold={d435_cfg.max_hold_time:.2f} s"
    )
  elif cfg.ball_observer == "d435":
    print(
      "Deployment parity: synchronized RGB-depth, "
      f"RGB fovy={d435_cfg.rgb_fovy_deg:.1f} deg, "
      f"depth ROI={d435_cfg.depth_roi_px} px, "
      f"hold={d435_cfg.max_hold_time:.2f} s"
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
        action, obs = _reset(
          model,
          data,
          bindings,
          metadata,
          keyboard.command,
          assembler,
          ball_observer,
          step_dt,
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

      policy_output = session.run(["actions"], assembler.policy_inputs(obs))[0]
      action = np.asarray(policy_output, dtype=np.float32)[0]
      if action.shape != (EXPECTED_ACTION_DIM,):
        raise RuntimeError(f"Policy returned invalid action shape {action.shape}.")
      if not np.all(np.isfinite(action)):
        raise RuntimeError("Policy returned NaN or Inf actions.")
      target = metadata.default_joint_pos + metadata.action_scale * action
      data.ctrl[bindings.actuator_ids] = target
      for _ in range(decimation):
        mujoco.mj_step(model, data)
        if viewer is not None:
          viewer.sync()

      policy_step += 1
      football_observation = ball_observer.observe(data)
      if detection_window is not None and d435_observer is not None:
        detection_window.update(
          d435_observer,
          football_observation,
          float(data.time),
        )
      terms = _observation_terms(
        data,
        bindings,
        metadata,
        keyboard.command,
        _phase(policy_step, step_dt, keyboard.command),
        action,
        football_observation,
      )
      obs = assembler.append(terms)
      if viewer is not None:
        remaining = step_dt - (time.perf_counter() - started)
        if remaining > 0.0:
          time.sleep(remaining)
  finally:
    ball_observer.close()
    if detection_window is not None:
      detection_window.close()
    if viewer is not None:
      viewer.close()


def main() -> None:
  """Parse command-line arguments and run football sim-to-sim."""
  run(tyro.cli(Sim2SimCfg, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
