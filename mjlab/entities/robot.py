import abc
from typing import Tuple
from mjlab.core import entity


class Robot(entity.Entity, abc.ABC):
  """Base class for robot entities."""

  @property
  @abc.abstractmethod
  def joint_names(self) -> Tuple[str, ...]:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def joint_stiffness(self) -> Tuple[float, ...]:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def joint_damping(self) -> Tuple[float, ...]:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def default_joint_pos_nominal(self) -> Tuple[float, ...]:
    raise NotImplementedError
