import abc
from dataclasses import dataclass, replace
from pathlib import Path
import fnmatch
from typing import Optional, TypeVar, Type, Dict, Tuple, Sequence

import numpy as np
from mjlab.core import entity
import mujoco

from mjlab.entities.robots import editors

T = TypeVar("T", bound="Robot")


@dataclass(frozen=True)
class RobotConfig:
  joints: Sequence[editors.Joint]
  actuators: Sequence[editors.Actuator]
  sensors: Sequence[editors.Sensor]
  keyframes: Sequence[editors.Keyframe]

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    for cfg in self.joints:
      cfg.edit_spec(spec)
    for cfg in self.sensors:
      cfg.edit_spec(spec)
    for cfg in self.keyframes:
      cfg.edit_spec(spec)

    # Add actuators in joint order.
    for joint in spec.joints:
      joint: mujoco.MjsJoint
      for act_cfg in self.actuators:
        if fnmatch.fnmatch(joint.name, act_cfg.joint_name):
          bound_cfg = replace(act_cfg, joint_name=joint.name)
          bound_cfg.edit_spec(spec)
          break


class Robot(entity.Entity, abc.ABC):
  """Abstract base class for robot entities.

  Subclasses must implement `joints` and `actuators` properties.
  """

  def __init__(self, spec: mujoco.MjSpec, config: RobotConfig | None = None):
    self._spec = spec
    self._config = config
    if config is not None:
      config.edit_spec(spec)

  @classmethod
  def from_file(
    cls: Type[T],
    xml_path: Path,
    config: Optional[RobotConfig] = None,
    assets: Optional[Dict[str, bytes]] = None,
  ) -> T:
    spec = mujoco.MjSpec.from_file(str(xml_path), assets=assets)
    return cls(spec, config=config)

  @classmethod
  def from_xml_str(
    cls: Type[T],
    xml_str: str,
    config: Optional[RobotConfig] = None,
    assets: Optional[Dict[str, bytes]] = None,
  ) -> T:
    spec = mujoco.MjSpec.from_string(xml_str, assets=assets)
    return cls(spec, config=config)

  @property
  @abc.abstractmethod
  def joints(self) -> Tuple[mujoco.MjsJoint, ...]:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def actuators(self) -> Tuple[mujoco.MjsActuator, ...]:
    raise NotImplementedError

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def joint_names(self) -> Tuple[str, ...]:
    return tuple([j.name for j in self.joints])

  def joint_stiffness(self) -> np.ndarray:
    return np.array([a.gainprm[0] for a in self.actuators])

  def joint_damping(self) -> np.ndarray:
    return np.array([-a.biasprm[2] for a in self.actuators])
