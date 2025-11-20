"""Wrappers for XML-defined actuators.

This module provides wrappers for actuators already defined in robot XML/MJCF files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.actuator import ActuatedType, Actuator, ActuatorCfg, ActuatorCmd

if TYPE_CHECKING:
  from mjlab.entity import Entity


class XmlActuator(Actuator):
  """Base class for XML-defined actuators."""

  def edit_spec(self, spec: mujoco.MjSpec, actuated_names: list[str]) -> None:
    # Filter to only joints that have corresponding XML actuators.
    filtered_actuated_ids = []
    filtered_actuated_names = []
    for i, actuated_name in enumerate(actuated_names):
      actuator = self._find_actuator_for_actuated_item(spec, actuated_name)
      if actuator is not None:
        self._mjs_actuators.append(actuator)
        filtered_actuated_ids.append(self._actuated_ids_list[i])
        filtered_actuated_names.append(actuated_name)

    if len(filtered_actuated_names) == 0:
      raise ValueError(
        f"No XML actuators found for any joints or any tendons matching the patterns. "
        f"Searched joints and tendons: {actuated_names}. "
        f"XML actuator config expects actuators to already exist in the XML."
      )

    # Update joint IDs and names to only include those with actuators.
    self._actuated_ids_list = filtered_actuated_ids
    self._actuated_names = filtered_actuated_names

  def _find_actuator_for_actuated_item(
    self, spec: mujoco.MjSpec, actuated_name: str
  ) -> mujoco.MjsActuator | None:
    """Find an actuator that targets the given joint."""
    for actuator in spec.actuators:
      if actuator.target == actuated_name:
        return actuator
    return None


@dataclass(kw_only=True)
class XmlPositionActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <position> actuators."""

  actuated_type: ActuatedType = ActuatedType.JOINT
  """Type of elements being actuated: 'joint'."""

  def build(
    self, entity: Entity, actuated_ids: list[int], actuated_names: list[str]
  ) -> XmlPositionActuator:
    return XmlPositionActuator(entity, actuated_ids, actuated_names)


class XmlPositionActuator(XmlActuator):
  """Wrapper for XML-defined <position> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.position_target


@dataclass(kw_only=True)
class XmlMotorActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <motor> actuators."""

  actuated_type: ActuatedType = ActuatedType.JOINT
  """Type of elements being actuated: 'joint'."""

  def build(
    self, entity: Entity, actuated_ids: list[int], actuated_names: list[str]
  ) -> XmlMotorActuator:
    return XmlMotorActuator(entity, actuated_ids, actuated_names)


class XmlMotorActuator(XmlActuator):
  """Wrapper for XML-defined <motor> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target


@dataclass(kw_only=True)
class XmlVelocityActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <velocity> actuators."""

  actuated_type: ActuatedType = ActuatedType.JOINT
  """Type of elements being actuated: 'joint'."""

  def build(
    self, entity: Entity, actuated_ids: list[int], actuated_names: list[str]
  ) -> XmlVelocityActuator:
    return XmlVelocityActuator(entity, actuated_ids, actuated_names)


class XmlVelocityActuator(XmlActuator):
  """Wrapper for XML-defined <velocity> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.velocity_target


@dataclass(kw_only=True)
class XmlMuscleActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <muscle> actuators."""

  actuated_type: ActuatedType = ActuatedType.TENDON
  """Type of elements being actuated: 'tendon'."""

  def build(
    self, entity: Entity, actuated_ids: list[int], actuated_names: list[str]
  ) -> XmlMuscleActuator:
    return XmlMuscleActuator(entity, actuated_ids, actuated_names)


class XmlMuscleActuator(XmlActuator):
  """Wrapper for XML-defined <muscle> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.excitation
