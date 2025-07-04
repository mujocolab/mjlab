import abc
from pathlib import Path
from typing import Optional, TypeVar, Type, Tuple, Dict, Tuple
from mjlab.core import entity
import mujoco

from mjlab.entities.robot_config import RobotConfig

T = TypeVar("T", bound="Robot")


class Robot(entity.Entity, abc.ABC):
  """Base class for robot entities."""

  def __init__(self, spec: mujoco.MjSpec, config: RobotConfig | None = None):
    self._config = config
    if config is not None:
      config.edit_spec(spec)
    super().__init__(spec)

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
  @abc.abstractmethod
  def joint_names(self) -> Tuple[str, ...]:
    raise NotImplementedError
