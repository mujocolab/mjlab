"""Wrappers for XML-defined actuators.

This module provides wrappers for actuators already defined in robot XML/MJCF files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd

if TYPE_CHECKING:
  from mjlab.entity import Entity

XmlActuatorCfgT = TypeVar("XmlActuatorCfgT", bound=ActuatorCfg)


class XmlActuator(Actuator[XmlActuatorCfgT], Generic[XmlActuatorCfgT]):
  """Base class for XML-defined actuators."""

  def __init__(
    self,
    cfg: XmlActuatorCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # Filter to only targets that have corresponding XML actuators.
    filtered_target_ids = []
    filtered_target_names = []
    for i, target_name in enumerate(target_names):
      actuator = self._find_actuator_for_target(spec, target_name)
      if actuator is not None:
        self._mjs_actuators.append(actuator)
        filtered_target_ids.append(self._target_ids_list[i])
        filtered_target_names.append(target_name)

    if len(filtered_target_names) == 0:
      raise ValueError(
        f"No XML actuators found for any targets matching the patterns. "
        f"Searched targets: {target_names}. "
        f"XML actuator config expects actuators to already exist in the XML."
      )

    # Update target IDs and names to only include those with actuators.
    self._target_ids_list = filtered_target_ids
    self._target_names = filtered_target_names

  def _find_actuator_for_target(
    self, spec: mujoco.MjSpec, target_name: str
  ) -> mujoco.MjsActuator | None:
    """Find an actuator that targets the given target (joint, tendon, or site)."""
    for actuator in spec.actuators:
      if actuator.target == target_name:
        return actuator
    return None


@dataclass(kw_only=True)
class XmlPositionActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <position> actuators."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> XmlPositionActuator:
    return XmlPositionActuator(self, entity, target_ids, target_names)


class XmlPositionActuator(XmlActuator[XmlPositionActuatorCfg]):
  """Wrapper for XML-defined <position> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.position_target


@dataclass(kw_only=True)
class XmlMotorActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <motor> actuators."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> XmlMotorActuator:
    return XmlMotorActuator(self, entity, target_ids, target_names)


class XmlMotorActuator(XmlActuator[XmlMotorActuatorCfg]):
  """Wrapper for XML-defined <motor> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target


@dataclass(kw_only=True)
class XmlVelocityActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <velocity> actuators."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> XmlVelocityActuator:
    return XmlVelocityActuator(self, entity, target_ids, target_names)


class XmlVelocityActuator(XmlActuator[XmlVelocityActuatorCfg]):
  """Wrapper for XML-defined <velocity> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.velocity_target


@dataclass(kw_only=True)
class XmlMuscleActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined <muscle> actuators."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> XmlMuscleActuator:
    return XmlMuscleActuator(self, entity, target_ids, target_names)


class XmlMuscleActuator(XmlActuator[XmlMuscleActuatorCfg]):
  """Wrapper for XML-defined <muscle> actuators."""

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target
