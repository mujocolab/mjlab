from dataclasses import dataclass
from typing import Sequence, Tuple, Protocol
import fnmatch

import numpy as np
import mujoco


class SpecEditor(Protocol):
  """Any object that knows how to edit an MjSpec."""

  def edit_spec(self, spec: mujoco.MjSpec) -> None: ...


@dataclass(frozen=True)
class RobotConfig:
  joints: Sequence[SpecEditor]
  actuators: Sequence[SpecEditor]
  sensors: Sequence[SpecEditor]
  keyframes: Sequence[SpecEditor]

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    for group in (self.joints, self.actuators, self.sensors, self.keyframes):
      for cfg in group:
        cfg.edit_spec(spec)


_SENSOR_TYPE_MAP = {
  "gyro": mujoco.mjtSensor.mjSENS_GYRO,
  "upvector": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
  "velocimeter": mujoco.mjtSensor.mjSENS_VELOCIMETER,
  "framequat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
  "framepos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
  "framelinvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
  "frameangvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
  "framezaxis": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
  "accelerometer": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
}

_SENSOR_OBJECT_TYPE_MAP = {
  "site": mujoco.mjtObj.mjOBJ_SITE,
}


@dataclass(frozen=True)
class Sensor:
  """Configuration for a sensor."""

  name: str
  sensor_type: str
  object_name: str
  object_type: str

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    spec.add_sensor(
      type=_SENSOR_TYPE_MAP[self.sensor_type],
      objtype=_SENSOR_OBJECT_TYPE_MAP[self.object_type],
      name=self.name,
      objname=self.object_name,
    )


def match_joints(spec: mujoco.MjSpec, pattern: str) -> Sequence[mujoco.MjsJoint]:
  matches = [j for j in spec.joints if fnmatch.fnmatch(j.name, pattern)]
  if not matches:
    raise ValueError(f"No joints matched pattern: {pattern}")
  return matches


@dataclass(frozen=True)
class Actuator:
  """Configuration for an actuator."""

  kp: float
  """Position gain."""
  kv: float
  """D gain."""
  joint_name: str
  """Name of the joint the actuator is applying torque to."""
  torque_limit: float | None = None
  """Torque limit. Assumed to be symmetric about zero. If None, no limit is applied."""
  inheritrange: float = 1.0
  """Sets the control range of the actuator to match the joint's target range."""

  def __post_init__(self):
    assert self.kp >= 0.0
    assert self.kv >= 0.0
    if self.torque_limit is not None:
      assert self.torque_limit > 0.0

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    for jnt in match_joints(spec, self.joint_name):
      act = spec.add_actuator(
        name=jnt.name,
        target=jnt.name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        inheritrange=self.inheritrange,
      )
      act.gainprm[0] = self.kp
      act.biasprm[1] = -self.kp
      act.biasprm[2] = -self.kv


@dataclass(frozen=True)
class Joint:
  """Configuration for a joint."""

  joint_name: str
  """Joint name."""
  damping: float = 0.0
  """Joint damping."""
  frictionloss: float = 0.0
  """Joint stiction."""
  armature: float = 0.0
  """Reflected inertia."""

  def __post_init__(self):
    assert self.damping >= 0.0
    assert self.armature >= 0.0
    assert self.frictionloss >= 0.0

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    for jnt in match_joints(spec, self.joint_name):
      jnt.damping = self.damping
      jnt.frictionloss = self.frictionloss
      jnt.armature = self.armature


@dataclass(frozen=True)
class Keyframe:
  """Configuration for a keyframe."""

  name: str
  """Name of the keyframe."""
  root_pos: np.ndarray
  """Root position in the world frame."""
  root_quat: np.ndarray
  """Root orientation as a (w, x, y, z) quaternion."""
  joint_angles: np.ndarray
  """Joint angles."""
  ctrl: np.ndarray | None = None
  """Actuator control signal."""

  @property
  def qpos(self) -> np.ndarray:
    return np.concatenate([self.root_pos, self.root_quat, self.joint_angles])

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    spec.add_key(name=self.name, qpos=self.qpos, ctrl=self.ctrl)


@dataclass(frozen=True)
class CollisionPair:
  """Configuration for a collision pair."""

  geom1: str
  geom2: str
  condim: int = 1
  friction: Sequence[float] | None = None
  solref: Sequence[float] | None = None
  solimp: Sequence[float] | None = None

  def full_name(self) -> str:
    return f"{self.geom1}__{self.geom2}"

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    pair = spec.add_pair(
      name=self.full_name(),
      geomname1=self.geom1,
      geomname2=self.geom2,
      condim=self.condim,
    )
    if self.friction is not None:
      for i in range(len(self.friction)):
        pair.friction[i] = self.friction[i]
    if self.solref is not None:
      pair.solref = self.solref
    if self.solimp is not None:
      for i in range(len(self.solimp)):
        pair.solimp[i] = self.solimp[i]


@dataclass(frozen=True)
class Skybox:
  rgb1: Tuple[float, float, float] = (0.3, 0.5, 0.7)
  rgb2: Tuple[float, float, float] = (0.1, 0.2, 0.3)
  width: int = 512
  height: int = 3072

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    spec.add_texture(
      type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
      rgb1=self.rgb1,
      rgb2=self.rgb2,
      width=self.width,
      height=self.height,
    )
