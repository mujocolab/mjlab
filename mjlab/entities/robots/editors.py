from dataclasses import dataclass
from typing import Sequence
import fnmatch
import numpy as np
import mujoco
from mjlab.core.editors import SpecEditor


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
class Sensor(SpecEditor):
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


@dataclass(frozen=True)
class Actuator(SpecEditor):
  """Configuration for an actuator."""

  kp: float
  kv: float
  joint_name: str
  torque_limit: float | None = None
  inheritrange: float = 1.0

  def __post_init__(self):
    assert self.kp >= 0.0
    assert self.kv >= 0.0
    if self.torque_limit is not None:
      assert self.torque_limit > 0.0

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    act = spec.add_actuator(
      name=self.joint_name,
      target=self.joint_name,
      trntype=mujoco.mjtTrn.mjTRN_JOINT,
      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
      inheritrange=self.inheritrange,
      forcerange=(-self.torque_limit, self.torque_limit),
    )
    act.gainprm[0] = self.kp
    act.biasprm[1] = -self.kp
    act.biasprm[2] = -self.kv


@dataclass(frozen=True)
class Joint(SpecEditor):
  """Configuration for a joint."""

  joint_name: str
  damping: float = 0.0
  frictionloss: float = 0.0
  armature: float = 0.0

  def __post_init__(self):
    assert self.damping >= 0.0
    assert self.armature >= 0.0
    assert self.frictionloss >= 0.0

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    for jnt in spec.joints:
      if fnmatch.fnmatch(jnt.name, self.joint_name):
        jnt.damping = self.damping
        jnt.frictionloss = self.frictionloss
        jnt.armature = self.armature


@dataclass(frozen=True)
class Keyframe(SpecEditor):
  """Configuration for a keyframe."""

  name: str
  root_pos: np.ndarray
  root_quat: np.ndarray
  joint_angles: np.ndarray
  ctrl: np.ndarray | None = None

  @property
  def qpos(self) -> np.ndarray:
    return np.concatenate([self.root_pos, self.root_quat, self.joint_angles])

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    spec.add_key(name=self.name, qpos=self.qpos, ctrl=self.ctrl)


@dataclass(frozen=True)
class CollisionPair(SpecEditor):
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
