from dataclasses import dataclass
from typing import Sequence

import numpy as np


# TODO(kevin): Add support for dampratio.
@dataclass(frozen=True)
class PDActuator:
  """A PD actuator."""

  kp: float
  """Position gain."""
  kv: float
  """D gain."""
  joint_name: str
  """Name of the joint the actuator is applying torque to."""
  armature: float = 0.0
  """Reflected inertia. Defaults to zero."""
  torque_limit: float | None = None
  """Torque limit. Assumed to be symmetric about zero. If None, no limit is applied."""
  frictionloss: float = 0.0
  """Stiction. Defaults to zero."""

  def __post_init__(self):
    assert self.kp >= 0.0
    assert self.kv >= 0.0
    assert self.armature >= 0.0
    if self.torque_limit is not None:
      assert self.torque_limit > 0.0
    assert self.frictionloss >= 0.0


@dataclass(frozen=True)
class CollisionPair:
  """A collision pair."""

  geom1: str
  geom2: str
  condim: int = 1
  friction: Sequence[float] | None = None
  solref: Sequence[float] | None = None
  solimp: Sequence[float] | None = None

  def full_name(self) -> str:
    return f"{self.geom1}__{self.geom2}"


@dataclass(frozen=True)
class Keyframe:
  """A keyframe."""

  name: str
  root_pos: np.ndarray
  root_quat: np.ndarray
  joint_angles: np.ndarray

  @staticmethod
  def initialize(
    name: str,
    root_pos: Sequence[float],
    root_quat: Sequence[float],
    joint_angles: Sequence[float],
  ) -> "Keyframe":
    return Keyframe(
      name=name,
      root_pos=np.array(root_pos),
      root_quat=np.array(root_quat),
      joint_angles=np.array(joint_angles),
    )

  @property
  def qpos(self) -> np.ndarray:
    return np.concatenate([self.root_pos, self.root_quat, self.joint_angles])


@dataclass(frozen=True)
class Sensor:
  """A sensor."""

  name: str
  sensor_type: str
  object_name: str
  object_type: str
