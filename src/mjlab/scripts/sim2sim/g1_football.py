"""Run the MJLab G1 football policy in native CPU MuJoCo."""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass, field
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
  D435Config,
  add_d435_camera,
  add_football_visual_material,
  make_ball_observer,
  world_to_yaw,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK_ID = "Mjlab-Velocity-Football-Flat-Unitree-G1"
FRAME_STACK = 5
TEMPORAL_HISTORY_LENGTH = 10
BALL_VISIBILITY_X_RANGE = (0.05, 1.00)
BALL_VISIBILITY_Y_RANGE = (-0.70, 0.70)
BALL_OBSERVATION_BIAS_RANGE = 0.10
BALL_OBSERVATION_FRAME_NOISE_RANGE = 0.0
BALL_OBSERVATION_MAX_DELAY_STEPS = 0
BALL_OBSERVATION_HOLD_PROBABILITY = 0.0
PHASE_PERIOD = 0.6
BALL_DISTURBANCE_INTERVAL_RANGE = (5.0, 6.0)
BALL_DISTURBANCE_LINEAR_VELOCITY_RANGE = (
  (-1.0, 1.0),
  (-1.0, 1.0),
  (-0.2, 0.2),
)
VELOCITY_SAMPLE_COUNT = 5
VELOCITY_PLOT_INDICES = (3, 4, 2, 1)
VELOCITY_PLOT_LABELS = (
  "Robot COM",
  "Ball",
  "Generated policy command",
  "Monotonic target",
)
VELOCITY_PLOT_COLORS = (
  (0.1, 0.8, 1.0),  # Robot center of mass.
  (1.0, 0.3, 0.3),  # Football.
  (1.0, 0.55, 0.0),  # Generated command sent to the policy.
  (0.1, 0.8, 0.2),  # Monotonic target; draw last so it stays visible.
)
POSITION_PLOT_COLORS = (VELOCITY_PLOT_COLORS[0], VELOCITY_PLOT_COLORS[1])
POSITION_PLOT_LABELS = ("Robot COM", "Ball")
RELATIVE_POSITION_PLOT_COLORS = (VELOCITY_PLOT_COLORS[0],)
RELATIVE_POSITION_PLOT_LABELS = ("Ball relative pelvis",)

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
B1_HISTORY_TERM_DIMS = {
  "ball_pos_b": 2,
  "ball_to_feet_vectors_b": 4,
  "ball_visible_mask": 1,
}
B1_HISTORY_OBSERVATION_NAMES = tuple(B1_HISTORY_TERM_DIMS)
B1_HISTORY_OBS_DIM = sum(B1_HISTORY_TERM_DIMS.values())
EXPECTED_ACTION_DIM = TERM_DIMS["actions"]

FloatArray = npt.NDArray[np.float32]
PlotSample = tuple[
  FloatArray,
  FloatArray,
  FloatArray,
  FloatArray,
  FloatArray,
  FloatArray,
  FloatArray,
  FloatArray,
  FloatArray,
]


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
    if not temporal and input_dim != EXPECTED_OBS_DIM:
      raise ValueError(
        f"Policy expects {input_dim} observations; this task requires "
        f"{EXPECTED_OBS_DIM}."
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
    if temporal and input_dim == TEMPORAL_OBS_DIM:
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
    self._term_dims = TEMPORAL_TERM_DIMS if self._temporal else TERM_DIMS
    self._observation_names = (
      metadata.observation_names
      if metadata is not None
      else EXPECTED_OBSERVATION_NAMES
    )
    self._all_names = (
      TEMPORAL_OBSERVATION_NAMES if self._temporal else EXPECTED_OBSERVATION_NAMES
    )
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
      raise ValueError(
        f"Observation terms must be ordered as {self._all_names}."
      )
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
        [
          self._history[name].flatten(FRAME_STACK)
          for name in self._observation_names
        ]
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
) -> tuple[mujoco.MjModel, float, int]:
  """Compile the nominal football play scene and apply its simulation options."""
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  scene = Scene(env_cfg.scene, device="cpu")
  add_d435_camera(scene.spec, d435_cfg or D435Config())
  add_football_visual_material(scene.spec)
  model = scene.compile()
  env_cfg.sim.mujoco.apply(model)
  return model, model.opt.timestep, env_cfg.decimation


def find_latest_policy(log_root: Path = Path("logs/rsl_rl")) -> Path:
  """Find the newest ONNX export compatible with the 520-value contract."""
  experiment_name = load_rl_cfg(TASK_ID).experiment_name
  experiment_dir = log_root / experiment_name
  policies = sorted(
    (path for path in experiment_dir.rglob("*.onnx") if path.is_file()),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
  )
  for path in policies:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] == EXPECTED_OBS_DIM:
      return path
  if not policies:
    raise FileNotFoundError(
      f"No ONNX policy found below {experiment_dir}. Run play once to export it, "
      "or pass --policy explicitly."
    )
  raise FileNotFoundError(
    f"No {EXPECTED_OBS_DIM}-input ONNX policy found below {experiment_dir}. "
    "The existing 535-input exports use the previous football observation contract."
  )


class KeyboardController:
  """Mutable velocity command controlled by the native MuJoCo viewer."""

  def __init__(self, command: FloatArray) -> None:
    self.command = command.copy()
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
      np.asarray([-0.25, -0.25, -1.0], dtype=np.float32),
      np.asarray([1.0, 0.25, 1.0], dtype=np.float32),
    )


class BallRelativeCommandGenerator:
  """Generate the policy command while preserving the user's final target."""

  def __init__(self) -> None:
    self.anchor = np.asarray([0.25, 0.0], dtype=np.float32)
    self.position_deadband = np.asarray([0.07, 0.06], dtype=np.float32)
    self.velocity_deadband = np.asarray([0.05, 0.05], dtype=np.float32)
    self.position_gain = np.asarray([0.3, 0.5], dtype=np.float32)
    self.velocity_gain = np.asarray([0.6, 0.6], dtype=np.float32)
    self.max_correction = np.asarray([0.4, 0.25], dtype=np.float32)
    self.filtered_position = np.zeros(2, dtype=np.float32)
    self.previous_filtered_position = np.zeros(2, dtype=np.float32)
    self.filtered_relative_velocity = np.zeros(2, dtype=np.float32)
    self.base_velocity = np.zeros(2, dtype=np.float32)
    self.reference = np.zeros(3, dtype=np.float32)
    self.initialized = False

  @staticmethod
  def _deadband(value: FloatArray, width: FloatArray) -> FloatArray:
    magnitude = np.maximum(np.abs(value) - width, 0.0)
    return (np.sign(value) * magnitude).astype(np.float32)

  def reset(self, user_command: FloatArray, ball_position: FloatArray) -> None:
    self.filtered_position = ball_position.astype(np.float32, copy=True)
    self.previous_filtered_position = self.filtered_position.copy()
    self.filtered_relative_velocity.fill(0.0)
    self.base_velocity = user_command[:2].astype(np.float32, copy=True)
    self.reference = user_command.astype(np.float32, copy=True)
    self.initialized = True

  def update(
    self,
    user_command: FloatArray,
    ball_position: FloatArray,
    dt: float,
  ) -> FloatArray:
    if not self.initialized:
      self.reset(user_command, ball_position)

    if np.any(np.abs(ball_position) > 1e-6):
      self.filtered_position += 0.2 * (ball_position - self.filtered_position)
      raw_velocity = (self.filtered_position - self.previous_filtered_position) / dt
      self.previous_filtered_position = self.filtered_position.copy()
      raw_speed = float(np.linalg.norm(raw_velocity))
      if raw_speed > 2.0:
        raw_velocity *= 2.0 / raw_speed
      self.filtered_relative_velocity += 0.1 * (
        raw_velocity - self.filtered_relative_velocity
      )

    position_error = self._deadband(
      self.filtered_position - self.anchor,
      self.position_deadband,
    )
    relative_velocity = self._deadband(
      self.filtered_relative_velocity,
      self.velocity_deadband,
    )
    correction = (
      self.position_gain * position_error + self.velocity_gain * relative_velocity
    )
    correction = np.clip(
      correction,
      -self.max_correction,
      self.max_correction,
    )
    base_difference = user_command[:2] - self.base_velocity
    base_norm = float(np.linalg.norm(base_difference))
    base_max_change = 0.4 * dt
    if base_norm > base_max_change:
      base_difference *= base_max_change / base_norm
    self.base_velocity += base_difference

    target = user_command.copy()
    target[:2] = self.base_velocity + correction
    target[:2] = np.clip(
      target[:2],
      np.asarray([-0.25, -0.25], dtype=np.float32),
      np.asarray([1.0, 0.25], dtype=np.float32),
    )

    difference = target[:2] - self.reference[:2]
    is_speeding_up = (self.reference[:2] * target[:2] >= 0.0) & (
      np.abs(target[:2]) > np.abs(self.reference[:2])
    )
    max_change = np.where(is_speeding_up, 0.8 * dt, 0.5 * dt)
    self.reference[:2] += np.clip(difference, -max_change, max_change)
    self.reference[2] = user_command[2]
    return self.reference.copy()


def _minimum_jerk(progress: float) -> float:
  """Return a minimum-jerk interpolation weight on the unit interval."""
  progress = float(np.clip(progress, 0.0, 1.0))
  return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


@dataclass(frozen=True)
class StopSkillCommandGeneratorCfg:
  """Configuration for the sim2sim-only keyboard deceleration skill."""

  enabled: bool = True
  maximum_velocity: float = 1.0
  rise_amplitude: float = 0.2
  rise_duration: float = 0.3
  fall_duration: float = 0.3
  trigger_window: int = 5
  acceleration_threshold: float = 1.0
  minimum_command_drop: float = 0.12
  persistence_frames: int = 2
  rearm_acceleration_threshold: float = 0.2

  def __post_init__(self) -> None:
    positive_values = (
      self.maximum_velocity,
      self.rise_duration,
      self.fall_duration,
      self.acceleration_threshold,
    )
    if any(value <= 0.0 for value in positive_values):
      raise ValueError(
        "Stop-skill velocity, durations, and threshold must be positive."
      )
    if self.rise_amplitude < 0.0:
      raise ValueError("Stop-skill rise amplitude must be non-negative.")
    if self.trigger_window < 1 or self.persistence_frames < 1:
      raise ValueError("Stop-skill window and persistence must be positive.")
    normalized_thresholds = (
      self.minimum_command_drop,
      self.rearm_acceleration_threshold,
    )
    if any(value < 0.0 for value in normalized_thresholds):
      raise ValueError("Stop-skill normalized thresholds must be non-negative.")


class StopSkillCommandGenerator:
  """Shape a rapid keyboard deceleration into a rise-and-fall reference."""

  IDLE = "IDLE"
  RISE = "RISE"
  FALL = "FALL"

  def __init__(self, cfg: StopSkillCommandGeneratorCfg) -> None:
    self.cfg = cfg
    self.history: deque[float] = deque(maxlen=cfg.trigger_window + 1)
    self.state = self.IDLE
    self.armed = True
    self.condition_count = 0
    self.elapsed = 0.0
    self.start_reference = np.zeros(3, dtype=np.float32)
    self.final_reference = np.zeros(3, dtype=np.float32)
    self.target_reference = np.zeros(3, dtype=np.float32)
    self.peak_reference = np.zeros(3, dtype=np.float32)
    self.green_at_peak = np.zeros(3, dtype=np.float32)
    self.reference = np.zeros(3, dtype=np.float32)

  @property
  def active(self) -> bool:
    return self.state != self.IDLE

  def _normalized_speed(self, command: FloatArray) -> float:
    return float(
      np.clip(
        np.linalg.norm(command[:2]) / self.cfg.maximum_velocity,
        0.0,
        1.0,
      )
    )

  def reset(self, keyboard_command: FloatArray) -> None:
    """Reset the trigger and references to the current keyboard command."""
    normalized_speed = self._normalized_speed(keyboard_command)
    self.history.clear()
    self.history.extend(normalized_speed for _ in range(self.cfg.trigger_window + 1))
    self.state = self.IDLE
    self.armed = True
    self.condition_count = 0
    self.elapsed = 0.0
    self.start_reference = keyboard_command.astype(np.float32, copy=True)
    self.final_reference = keyboard_command.astype(np.float32, copy=True)
    self.target_reference = keyboard_command.astype(np.float32, copy=True)
    self.peak_reference = keyboard_command.astype(np.float32, copy=True)
    self.green_at_peak = keyboard_command.astype(np.float32, copy=True)
    self.reference = keyboard_command.astype(np.float32, copy=True)

  def _triggered(
    self,
    keyboard_command: FloatArray,
    dt: float,
  ) -> bool:
    normalized_command = self._normalized_speed(keyboard_command)
    old_command = self.history[0]
    window_drop = old_command - normalized_command
    window_acceleration = (normalized_command - old_command) / (
      self.cfg.trigger_window * dt
    )
    condition = (
      window_acceleration < -self.cfg.acceleration_threshold
      and window_drop > self.cfg.minimum_command_drop
    )
    self.condition_count = self.condition_count + 1 if condition else 0
    triggered = (
      self.armed
      and not self.active
      and self.condition_count >= self.cfg.persistence_frames
    )
    if triggered:
      self.armed = False
      self.condition_count = 0
    elif (
      not self.active
      and abs(window_acceleration) < self.cfg.rearm_acceleration_threshold
    ):
      self.armed = True
    self.history.append(normalized_command)
    return triggered

  def _green_reference(self, elapsed: float) -> FloatArray:
    total_duration = self.cfg.rise_duration + self.cfg.fall_duration
    weight = _minimum_jerk(elapsed / total_duration)
    output = self.start_reference + weight * (
      self.final_reference - self.start_reference
    )
    return output.astype(np.float32)

  def _start(self, keyboard_command: FloatArray) -> None:
    self.state = self.RISE
    self.elapsed = 0.0
    self.start_reference = self.reference.copy()
    self.final_reference = keyboard_command.astype(np.float32, copy=True)
    planar_speed = float(np.linalg.norm(self.start_reference[:2]))
    if planar_speed > 1e-6:
      direction = self.start_reference[:2] / planar_speed
    else:
      direction = np.zeros(2, dtype=np.float32)
    peak_speed = min(
      planar_speed + self.cfg.rise_amplitude,
      self.cfg.maximum_velocity,
    )
    self.peak_reference = self.start_reference.copy()
    self.peak_reference[:2] = direction * peak_speed
    self.green_at_peak = self._green_reference(self.cfg.rise_duration)

  def update(
    self,
    keyboard_command: FloatArray,
    dt: float,
  ) -> FloatArray:
    """Update and return the physical velocity reference sent to the policy."""
    if dt <= 0.0:
      raise ValueError("Stop-skill update period must be positive.")
    if not self.history:
      self.reset(keyboard_command)

    triggered = self._triggered(
      keyboard_command,
      dt,
    )
    if triggered:
      self._start(keyboard_command)

    if not self.active:
      if self.condition_count > 0:
        return self.reference.copy()
      self.final_reference = keyboard_command.astype(np.float32, copy=True)
      self.target_reference = keyboard_command.astype(np.float32, copy=True)
      self.reference = keyboard_command.astype(np.float32, copy=True)
      return self.reference.copy()

    self.elapsed += dt
    green_reference = self._green_reference(self.elapsed)
    self.target_reference = green_reference
    if self.elapsed <= self.cfg.rise_duration:
      weight = _minimum_jerk(self.elapsed / self.cfg.rise_duration)
      self.reference = self.start_reference + weight * (
        self.peak_reference - self.start_reference
      )
    else:
      fall_elapsed = self.elapsed - self.cfg.rise_duration
      weight = _minimum_jerk(fall_elapsed / self.cfg.fall_duration)
      self.state = self.FALL
      self.reference = green_reference + (self.peak_reference - self.green_at_peak) * (
        1.0 - weight
      )
      if fall_elapsed >= self.cfg.fall_duration:
        self.state = self.IDLE
        self.reference = self.target_reference.copy()

    self.reference[2] = keyboard_command[2]
    self.target_reference[2] = keyboard_command[2]
    return self.reference.astype(np.float32, copy=True)


class BallVelocityDisturbance:
  """Periodically add a world-frame linear velocity kick to the football."""

  def __init__(self, ball_dof_adr: int, rng: np.random.Generator | None = None) -> None:
    self._ball_dof_adr = ball_dof_adr
    self._rng = rng or np.random.default_rng()
    self.next_trigger_time = 0.0

  def reset(self, current_time: float) -> None:
    self.next_trigger_time = current_time + self._rng.uniform(
      *BALL_DISTURBANCE_INTERVAL_RANGE
    )

  def update(self, data: mujoco.MjData) -> FloatArray | None:
    """Apply a due disturbance and return its sampled XYZ velocity delta."""
    if data.time < self.next_trigger_time:
      return None
    delta = np.asarray(
      [self._rng.uniform(*bounds) for bounds in BALL_DISTURBANCE_LINEAR_VELOCITY_RANGE],
      dtype=np.float32,
    )
    data.qvel[self._ball_dof_adr : self._ball_dof_adr + 3] += delta
    self.reset(float(data.time))
    return delta


class PerturbedBallObserver:
  """Wrap a visual observer with real-like position bias and jitter."""

  def __init__(
    self,
    observer: Any,
    seed: int | None = None,
    bias_range: float = BALL_OBSERVATION_BIAS_RANGE,
    frame_noise_range: float = BALL_OBSERVATION_FRAME_NOISE_RANGE,
    max_delay_steps: int = BALL_OBSERVATION_MAX_DELAY_STEPS,
    hold_probability: float = BALL_OBSERVATION_HOLD_PROBABILITY,
  ) -> None:
    if bias_range < 0.0 or frame_noise_range < 0.0:
      raise ValueError("Ball observation disturbance ranges must be non-negative.")
    if max_delay_steps < 0:
      raise ValueError("Ball observation delay must be non-negative.")
    if not 0.0 <= hold_probability <= 1.0:
      raise ValueError("Ball observation hold probability must be in [0, 1].")
    self._observer = observer
    self._rng = np.random.default_rng(seed)
    self._bias_range = bias_range
    self._frame_noise_range = frame_noise_range
    self._max_delay_steps = max_delay_steps
    self._hold_probability = hold_probability
    self._bias = np.zeros(2, dtype=np.float32)
    self._delay_steps = 0
    self._history: deque[tuple[FloatArray, FloatArray]] = deque(maxlen=3)
    self._last: tuple[FloatArray, FloatArray] | None = None

  def reset(self) -> None:
    self._observer.reset()
    self._bias = self._rng.uniform(
      -self._bias_range, self._bias_range, size=2
    ).astype(np.float32)
    self._delay_steps = (
      int(self._rng.integers(0, self._max_delay_steps + 1))
      if self._max_delay_steps > 0
      else 0
    )
    self._history.clear()
    self._last = None

  def observe(self, data: mujoco.MjData) -> tuple[FloatArray, FloatArray]:
    ball_pos, feet_to_ball = self._observer.observe(data)
    ball_pos = np.asarray(ball_pos, dtype=np.float32)
    feet_to_ball = np.asarray(feet_to_ball, dtype=np.float32)
    if np.any(np.abs(ball_pos) > 1e-6):
      delta = self._bias + self._rng.uniform(
        -self._frame_noise_range,
        self._frame_noise_range,
        size=2,
      )
      ball_pos = ball_pos + delta
      feet_to_ball = feet_to_ball.reshape(-1, 2) + delta
      feet_to_ball = feet_to_ball.reshape(-1).astype(np.float32)
    current = (ball_pos.copy(), feet_to_ball.copy())
    self._history.append(current)
    if self._last is not None and self._rng.random() < self._hold_probability:
      return self._last
    index = max(0, len(self._history) - 1 - self._delay_steps)
    output = self._history[index]
    self._last = (output[0].copy(), output[1].copy())
    return self._last

  def close(self) -> None:
    self._observer.close()


def compute_planar_velocities(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  bindings: ModelBindings,
  user_command: FloatArray,
  monotonic_target: FloatArray,
  policy_reference: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
  """Return command and physical XY velocities in the robot yaw frame."""
  mujoco.mj_subtreeVel(model, data)
  root_quat = data.xquat[bindings.root_body_id]
  robot_com_vel = world_to_yaw(data.subtree_linvel[bindings.root_body_id], root_quat)[
    :2
  ]
  ball_vel_w = data.qvel[bindings.ball_dof_adr : bindings.ball_dof_adr + 3]
  ball_vel = world_to_yaw(ball_vel_w, root_quat)[:2]
  return (
    user_command[:2].astype(np.float32, copy=True),
    monotonic_target[:2].astype(np.float32, copy=True),
    policy_reference[:2].astype(np.float32, copy=True),
    robot_com_vel,
    ball_vel,
  )


def compute_planar_positions(
  data: mujoco.MjData,
  bindings: ModelBindings,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
  """Return world positions and yaw-frame ball positions relative to COM/pelvis."""
  robot_com_pos_w = data.subtree_com[bindings.root_body_id]
  robot_pelvis_pos_w = data.xpos[bindings.root_body_id]
  ball_pos_w = data.xpos[bindings.ball_body_id]
  root_quat = data.xquat[bindings.root_body_id]
  ball_relative_com = world_to_yaw(
    ball_pos_w - robot_com_pos_w,
    root_quat,
  )[:2]
  ball_relative_pelvis = world_to_yaw(
    ball_pos_w - robot_pelvis_pos_w,
    root_quat,
  )[:2]
  return (
    robot_com_pos_w[:2].astype(np.float32, copy=True),
    ball_pos_w[:2].astype(np.float32, copy=True),
    ball_relative_com,
    ball_relative_pelvis,
  )


class VelocityPlotter:
  """Maintain native MuJoCo figures for planar velocity and position tracking."""

  def __init__(
    self,
    history_seconds: float,
    sample_dt: float,
    *,
    record_full_history: bool = False,
  ) -> None:
    if history_seconds <= 0.0:
      raise ValueError("Velocity plot history must be positive.")
    if sample_dt <= 0.0:
      raise ValueError("Velocity plot sample period must be positive.")

    probe = mujoco.MjvFigure()
    max_points = probe.linedata.shape[1] // 2
    history_points = max(2, round(history_seconds / sample_dt))
    self._history_points = min(history_points, max_points)
    self._samples: deque[PlotSample] = deque(maxlen=self._history_points)
    self._record_full_history = record_full_history
    self._recorded_samples: list[PlotSample] = []
    self._recorded_generator_states: list[str] = []
    self._figures = (
      self._make_figure(
        "Yaw-frame X velocity (m/s)",
        VELOCITY_PLOT_LABELS,
        VELOCITY_PLOT_COLORS,
      ),
      self._make_figure(
        "Yaw-frame Y velocity (m/s)",
        VELOCITY_PLOT_LABELS,
        VELOCITY_PLOT_COLORS,
      ),
      self._make_figure(
        "World X position (m)",
        POSITION_PLOT_LABELS,
        POSITION_PLOT_COLORS,
      ),
      self._make_figure(
        "World Y position (m)",
        POSITION_PLOT_LABELS,
        POSITION_PLOT_COLORS,
      ),
      self._make_figure(
        "Ball relative COM X (m)",
        RELATIVE_POSITION_PLOT_LABELS,
        RELATIVE_POSITION_PLOT_COLORS,
      ),
      self._make_figure(
        "Ball relative COM Y (m)",
        RELATIVE_POSITION_PLOT_LABELS,
        RELATIVE_POSITION_PLOT_COLORS,
      ),
    )

  def _make_figure(
    self,
    title: str,
    labels: tuple[str, ...],
    colors: tuple[tuple[float, float, float], ...],
  ) -> mujoco.MjvFigure:
    figure = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(figure)
    figure.title = title
    figure.flg_extend = 1
    figure.gridsize[:] = (3, 4)
    figure.figurergba[3] = 0.65
    for line, (label, color) in enumerate(zip(labels, colors, strict=True)):
      figure.linename[line] = label
      figure.linergb[line] = color
    return figure

  def reset(self) -> None:
    """Clear all plotted samples."""
    self._samples.clear()
    self._recorded_samples.clear()
    self._recorded_generator_states.clear()
    for figure in self._figures:
      figure.linepnt[:] = 0

  def append(
    self,
    user_command_velocity: FloatArray,
    monotonic_target_velocity: FloatArray,
    policy_reference_velocity: FloatArray,
    robot_com_velocity: FloatArray,
    ball_velocity: FloatArray,
    robot_com_position: FloatArray,
    ball_position: FloatArray,
    ball_relative_com_position: FloatArray,
    ball_relative_pelvis_position: FloatArray,
    *,
    generator_state: str = StopSkillCommandGenerator.IDLE,
  ) -> None:
    """Append one XY velocity and position sample."""
    velocities = (
      user_command_velocity,
      monotonic_target_velocity,
      policy_reference_velocity,
      robot_com_velocity,
      ball_velocity,
    )
    positions = (
      robot_com_position,
      ball_position,
      ball_relative_com_position,
      ball_relative_pelvis_position,
    )
    values = velocities + positions
    if any(value.shape != (2,) for value in values):
      raise ValueError("Kinematics plot samples must contain XY values.")
    if not all(np.all(np.isfinite(value)) for value in values):
      return
    sample = tuple(value.copy() for value in values)
    self._samples.append(sample)
    if self._record_full_history:
      self._recorded_samples.append(sample)
      self._recorded_generator_states.append(generator_state)
    self._write_figures()

  def _write_figures(self) -> None:
    sample_count = len(self._samples)
    if sample_count == 0:
      return
    samples = np.asarray(self._samples, dtype=np.float32)

    for axis in range(2):
      self._write_series(
        self._figures[axis],
        samples[:, VELOCITY_PLOT_INDICES, axis],
      )
      self._write_series(
        self._figures[axis + 2],
        samples[
          :,
          VELOCITY_SAMPLE_COUNT : VELOCITY_SAMPLE_COUNT + len(POSITION_PLOT_LABELS),
          axis,
        ],
      )
      self._write_series(
        self._figures[axis + 4],
        samples[:, -len(RELATIVE_POSITION_PLOT_LABELS) :, axis],
      )

  @staticmethod
  def _write_series(figure: mujoco.MjvFigure, values: np.ndarray) -> None:
    sample_count, line_count = values.shape
    lo = min(float(np.min(values)), 0.0)
    hi = max(float(np.max(values)), 0.0)
    span = max(hi - lo, 0.2)
    padding = 0.15 * span
    figure.range[1][0] = lo - padding
    figure.range[1][1] = hi + padding

    for line in range(line_count):
      figure.linepnt[line] = sample_count
      for index in range(sample_count):
        figure.linedata[line][2 * index] = float(index - sample_count + 1)
        figure.linedata[line][2 * index + 1] = float(values[index, line])

  def set_viewer_figures(self, viewer: Any) -> None:
    """Place kinematics figures in a right-side three-by-two grid."""
    viewport = viewer.viewport
    plot_width = max(1, int(viewport.width * 0.24))
    plot_height = max(1, int(viewport.height * 0.24))
    left = viewport.left + viewport.width - 2 * plot_width
    viewports = (
      mujoco.MjrRect(
        left=left,
        bottom=viewport.bottom,
        width=plot_width,
        height=plot_height,
      ),
      mujoco.MjrRect(
        left=left + plot_width,
        bottom=viewport.bottom,
        width=plot_width,
        height=plot_height,
      ),
      mujoco.MjrRect(
        left=left,
        bottom=viewport.bottom + plot_height,
        width=plot_width,
        height=plot_height,
      ),
      mujoco.MjrRect(
        left=left + plot_width,
        bottom=viewport.bottom + plot_height,
        width=plot_width,
        height=plot_height,
      ),
      mujoco.MjrRect(
        left=left,
        bottom=viewport.bottom + 2 * plot_height,
        width=plot_width,
        height=plot_height,
      ),
      mujoco.MjrRect(
        left=left + plot_width,
        bottom=viewport.bottom + 2 * plot_height,
        width=plot_width,
        height=plot_height,
      ),
    )
    viewer.set_figures(list(zip(viewports, self._figures, strict=True)))

  def save(self, output_path: Path, sample_dt: float) -> Path:
    """Save the complete post-reset kinematics history as a static curve plot."""
    if not self._recorded_samples:
      raise RuntimeError("Cannot save an empty kinematics plot.")
    if output_path.suffix == "":
      output_path = output_path.with_suffix(".png")
    if output_path.suffix.lower() not in {".pdf", ".png", ".svg"}:
      raise ValueError("Velocity plot output must use .png, .pdf, or .svg.")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(self._recorded_samples, dtype=np.float32)
    times = np.arange(len(samples), dtype=np.float32) * sample_dt
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for axis_index, axis_name in enumerate(("X", "Y")):
      velocity_axis = axes[2, axis_index]
      for series_index, (sample_index, label) in enumerate(
        zip(
          VELOCITY_PLOT_INDICES,
          VELOCITY_PLOT_LABELS,
          strict=True,
        )
      ):
        velocity_axis.plot(
          times,
          samples[:, sample_index, axis_index],
          color=VELOCITY_PLOT_COLORS[series_index],
          label=label,
          linewidth=1.8 if label == "Monotonic target" else 1.2,
        )
      velocity_axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
      velocity_axis.set_ylabel(f"Yaw-frame {axis_name} velocity (m/s)")
      velocity_axis.grid(alpha=0.25)
      velocity_axis.legend(loc="upper right")
      velocity_axis.set_xlabel("Time (s)")

      position_axis = axes[0, axis_index]
      for series_index, label in enumerate(POSITION_PLOT_LABELS):
        position_axis.plot(
          times,
          samples[:, VELOCITY_SAMPLE_COUNT + series_index, axis_index],
          color=POSITION_PLOT_COLORS[series_index],
          label=label,
          linewidth=1.2,
        )
      position_axis.set_ylabel(f"World {axis_name} position (m)")
      position_axis.grid(alpha=0.25)
      position_axis.legend(loc="upper right")

      relative_axis = axes[1, axis_index]
      relative_axis.plot(
        times,
        samples[:, -1, axis_index],
        color=RELATIVE_POSITION_PLOT_COLORS[0],
        label=RELATIVE_POSITION_PLOT_LABELS[0],
        linewidth=1.2,
      )
      relative_axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
      relative_axis.set_ylabel(f"Yaw-frame relative {axis_name} (m)")
      relative_axis.grid(alpha=0.25)
      relative_axis.legend(loc="upper right")
    figure.suptitle("Robot COM and football planar kinematics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path

  def save_csv(self, output_path: Path, sample_dt: float) -> Path:
    """Save the complete post-reset planar kinematics history as CSV."""
    if not self._recorded_samples:
      raise RuntimeError("Cannot save empty kinematics data.")
    if output_path.suffix == "":
      output_path = output_path.with_suffix(".csv")
    if output_path.suffix.lower() != ".csv":
      raise ValueError("Kinematics data output must use .csv.")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
      "time",
      "generator_state",
      "user_command_vx",
      "user_command_vy",
      "monotonic_target_vx",
      "monotonic_target_vy",
      "policy_reference_vx",
      "policy_reference_vy",
      "robot_com_vx",
      "robot_com_vy",
      "ball_vx",
      "ball_vy",
      "robot_com_x",
      "robot_com_y",
      "ball_x",
      "ball_y",
      "ball_relative_com_x",
      "ball_relative_com_y",
      "ball_relative_pelvis_x",
      "ball_relative_pelvis_y",
    )
    with output_path.open("w", newline="") as file:
      writer = csv.writer(file)
      writer.writerow(header)
      for index, (sample, generator_state) in enumerate(
        zip(
          self._recorded_samples,
          self._recorded_generator_states,
          strict=True,
        )
      ):
        writer.writerow(
          (
            index * sample_dt,
            generator_state,
            *sample[0],
            *sample[1],
            *sample[2],
            *sample[3],
            *sample[4],
            *sample[5],
            *sample[6],
            *sample[7],
            *sample[8],
          )
        )
    return output_path


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
    "actions": np.clip(last_action, -10.0, 10.0).astype(np.float32),
    "ball_pos_b": ball_pos,
    "ball_to_feet_vectors_b": feet_to_ball,
  }
  if metadata.is_temporal:
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
  command_generator: BallRelativeCommandGenerator | None,
  stop_skill_generator: StopSkillCommandGenerator | None,
  step_dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
  mujoco.mj_resetDataKeyframe(model, data, bindings.init_key_id)
  data.ctrl[bindings.actuator_ids] = metadata.default_joint_pos
  mujoco.mj_forward(model, data)
  ball_observer.reset()
  action = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)
  football_observation = ball_observer.observe(data)
  policy_reference = user_command.copy()
  monotonic_target = user_command.copy()
  if command_generator is not None:
    command_generator.reset(user_command, football_observation[0])
    policy_reference = command_generator.update(
      user_command,
      football_observation[0],
      step_dt,
    )
  if stop_skill_generator is not None:
    stop_skill_generator.reset(user_command)
    policy_reference = stop_skill_generator.reference.copy()
    monotonic_target = stop_skill_generator.target_reference.copy()
  terms = _observation_terms(
    data,
    bindings,
    metadata,
    policy_reference,
    _phase(0, step_dt, policy_reference),
    action,
    football_observation,
  )
  return action, assembler.reset(terms), policy_reference, monotonic_target


@dataclass(frozen=True)
class Sim2SimCfg:
  """Command-line configuration for native MuJoCo football deployment."""

  policy: Path | None = None
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
  ball_observer: Literal["d435", "mujoco"] = "d435"
  ball_observation_disturbance: bool = False
  ball_observation_seed: int | None = None
  ball_relative_command_generator: bool = False
  stop_skill: StopSkillCommandGeneratorCfg = field(
    default_factory=StopSkillCommandGeneratorCfg
  )
  yolo_model: Path | None = None
  yolo_confidence: float = 0.25
  ball_hold_time: float = 0.5
  ball_velocity_disturbance: bool = False
  ball_velocity_seed: int | None = None
  show_velocity_plot: bool = True
  velocity_plot_history: float = 10.0
  velocity_plot_output: Path | None = None
  kinematics_data_output: Path | None = None


def run(cfg: Sim2SimCfg) -> None:
  """Load a policy and execute it in native MuJoCo."""
  stop_skill_enabled = cfg.stop_skill.enabled
  if cfg.ball_relative_command_generator and stop_skill_enabled:
    stop_skill_enabled = False
    print(
      "Stop-skill command generator: disabled because the ball-relative "
      "command generator takes precedence."
    )
  policy_path = (cfg.policy or find_latest_policy(cfg.log_root)).resolve()
  if not policy_path.is_file():
    raise FileNotFoundError(f"ONNX policy does not exist: {policy_path}")
  session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
  metadata = PolicyMetadata.from_session(session)
  d435_cfg = D435Config(
    confidence_threshold=cfg.yolo_confidence,
    max_hold_time=cfg.ball_hold_time,
  )
  model, timestep, decimation = build_model(d435_cfg)
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
  if cfg.ball_observation_disturbance:
    ball_observer = PerturbedBallObserver(
      ball_observer,
      seed=cfg.ball_observation_seed,
    )
  assembler = ObservationAssembler(metadata)
  command = np.asarray(
    [cfg.command_x, cfg.command_y, cfg.command_yaw], dtype=np.float32
  )
  keyboard = KeyboardController(command)
  step_dt = timestep * decimation
  command_generator = (
    BallRelativeCommandGenerator() if cfg.ball_relative_command_generator else None
  )
  stop_skill_generator = (
    StopSkillCommandGenerator(cfg.stop_skill) if stop_skill_enabled else None
  )
  action, obs, policy_reference, monotonic_target = _reset(
    model,
    data,
    bindings,
    metadata,
    keyboard.command,
    assembler,
    ball_observer,
    command_generator,
    stop_skill_generator,
    step_dt,
  )
  ball_disturbance = (
    BallVelocityDisturbance(
      bindings.ball_dof_adr,
      rng=np.random.default_rng(cfg.ball_velocity_seed),
    )
    if cfg.ball_velocity_disturbance
    else None
  )
  if ball_disturbance is not None:
    ball_disturbance.reset(float(data.time))
  velocity_plotter = (
    VelocityPlotter(
      cfg.velocity_plot_history,
      step_dt,
      record_full_history=(
        cfg.velocity_plot_output is not None or cfg.kinematics_data_output is not None
      ),
    )
    if (cfg.show_velocity_plot and not cfg.headless)
    or cfg.velocity_plot_output is not None
    or cfg.kinematics_data_output is not None
    else None
  )
  if velocity_plotter is not None:
    velocity_plotter.append(
      *compute_planar_velocities(
        model,
        data,
        bindings,
        keyboard.command,
        monotonic_target,
        policy_reference,
      ),
      *compute_planar_positions(data, bindings),
      generator_state=(
        stop_skill_generator.state
        if stop_skill_generator is not None
        else StopSkillCommandGenerator.IDLE
      ),
    )
  policy_step = 0
  total_policy_steps = max(0, int(cfg.duration / step_dt))
  stop_command_applied = False

  viewer: Any = None
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
    if velocity_plotter is not None:
      velocity_plotter.set_viewer_figures(viewer)

  print(f"Policy: {policy_path}")
  print(
    f"Native MuJoCo: dt={timestep:.3f}s, decimation={decimation}, "
    f"policy_rate={1.0 / step_dt:.1f}Hz"
  )
  print(
    f"Football observation: source={cfg.ball_observer}, viewer_camera={cfg.camera_view}"
  )
  if cfg.ball_observation_disturbance:
    print(
      "Ball observation disturbance: "
      f"episode_bias=+-{BALL_OBSERVATION_BIAS_RANGE:.2f}m, "
      f"frame_noise=+-{BALL_OBSERVATION_FRAME_NOISE_RANGE:.2f}m, "
      f"delay=0-{BALL_OBSERVATION_MAX_DELAY_STEPS} steps, "
      f"hold_probability={BALL_OBSERVATION_HOLD_PROBABILITY:.2f}"
    )
  if command_generator is not None:
    print(
      "Ball-relative command generator: enabled "
      "(anchor=[0.25, 0.0], Kp=[0.3, 0.5], Kv=[0.6, 0.6], a_max=0.8)"
    )
  if stop_skill_generator is not None:
    print(
      "Stop-skill command generator: enabled "
      f"(rise={cfg.stop_skill.rise_amplitude:.3f}m/s/"
      f"{cfg.stop_skill.rise_duration:.3f}s, "
      f"fall={cfg.stop_skill.fall_duration:.3f}s)"
    )
  if ball_disturbance is not None:
    print(
      "Ball velocity disturbance: interval=5-6s, "
      "delta_vx=+-1.0, delta_vy=+-1.0, delta_vz=+-0.2 m/s"
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
        action, obs, policy_reference, monotonic_target = _reset(
          model,
          data,
          bindings,
          metadata,
          keyboard.command,
          assembler,
          ball_observer,
          command_generator,
          stop_skill_generator,
          step_dt,
        )
        if ball_disturbance is not None:
          ball_disturbance.reset(float(data.time))
        if velocity_plotter is not None:
          velocity_plotter.reset()
          velocity_plotter.append(
            *compute_planar_velocities(
              model,
              data,
              bindings,
              keyboard.command,
              monotonic_target,
              policy_reference,
            ),
            *compute_planar_positions(data, bindings),
            generator_state=(
              stop_skill_generator.state
              if stop_skill_generator is not None
              else StopSkillCommandGenerator.IDLE
            ),
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

      if ball_disturbance is not None:
        ball_disturbance.update(data)

      if velocity_plotter is not None:
        velocity_plotter.append(
          *compute_planar_velocities(
            model,
            data,
            bindings,
            keyboard.command,
            monotonic_target,
            policy_reference,
          ),
          *compute_planar_positions(data, bindings),
          generator_state=(
            stop_skill_generator.state
            if stop_skill_generator is not None
            else StopSkillCommandGenerator.IDLE
          ),
        )

      policy_step += 1
      football_observation = ball_observer.observe(data)
      policy_reference = keyboard.command.copy()
      monotonic_target = keyboard.command.copy()
      if command_generator is not None:
        policy_reference = command_generator.update(
          keyboard.command,
          football_observation[0],
          step_dt,
        )
      if stop_skill_generator is not None:
        policy_reference = stop_skill_generator.update(
          keyboard.command,
          step_dt,
        )
        monotonic_target = stop_skill_generator.target_reference.copy()
      terms = _observation_terms(
        data,
        bindings,
        metadata,
        policy_reference,
        _phase(policy_step, step_dt, policy_reference),
        action,
        football_observation,
      )
      obs = assembler.append(terms)
      if viewer is not None:
        if velocity_plotter is not None and viewer.is_running():
          velocity_plotter.set_viewer_figures(viewer)
        remaining = step_dt - (time.perf_counter() - started)
        if remaining > 0.0:
          time.sleep(remaining)
  finally:
    if velocity_plotter is not None and cfg.velocity_plot_output is not None:
      saved_path = velocity_plotter.save(cfg.velocity_plot_output, step_dt)
      print(f"Kinematics plot saved to: {saved_path}")
    if velocity_plotter is not None and cfg.kinematics_data_output is not None:
      saved_path = velocity_plotter.save_csv(
        cfg.kinematics_data_output,
        step_dt,
      )
      print(f"Kinematics data saved to: {saved_path}")
    ball_observer.close()
    if viewer is not None:
      viewer.close()


def main() -> None:
  """Parse command-line arguments and run football sim-to-sim."""
  run(tyro.cli(Sim2SimCfg, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
