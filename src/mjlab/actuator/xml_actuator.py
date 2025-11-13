"""Wrappers for XML-defined actuators.

This module provides wrappers for actuators already defined in robot XML/MJCF files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd

if TYPE_CHECKING:
  from mjlab.entity import Entity


class XmlActuator(Actuator):
  """Base class for XML-defined actuators."""

  def edit_spec(self, spec: mujoco.MjSpec, joint_names: list[str]) -> None:
    for joint_name in joint_names:
      actuator = self._find_actuator_for_joint(spec, joint_name)
      if actuator is None:
        raise ValueError(
          f"No XML actuator found for joint '{joint_name}'. "
          f"XML actuator config expects actuators to already exist in the XML."
        )
      self._mjs_actuators.append(actuator)

  def _find_actuator_for_joint(
    self, spec: mujoco.MjSpec, joint_name: str
  ) -> mujoco.MjsActuator | None:
    """Find an actuator that targets the given joint."""
    for actuator in spec.actuators:
      if actuator.target == joint_name:
        return actuator
    return None


@dataclass(kw_only=True)
class XmlPositionActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <position> actuators."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> XmlPositionActuator:
    return XmlPositionActuator(entity, joint_ids, joint_names)


class XmlPositionActuator(XmlActuator):
  """Wrapper for XML-defined <position> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.position_target


@dataclass(kw_only=True)
class XmlMotorActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <motor> actuators."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> XmlMotorActuator:
    return XmlMotorActuator(entity, joint_ids, joint_names)


class XmlMotorActuator(XmlActuator):
  """Wrapper for XML-defined <motor> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target


@dataclass(kw_only=True)
class XmlVelocityActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <velocity> actuators."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> XmlVelocityActuator:
    return XmlVelocityActuator(entity, joint_ids, joint_names)


class XmlVelocityActuator(XmlActuator):
  """Wrapper for XML-defined <velocity> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.velocity_target
