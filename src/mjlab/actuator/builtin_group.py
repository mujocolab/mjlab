from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuator,
  BuiltinMuscleActuator,
  BuiltinPositionActuator,
  BuiltinVelocityActuator,
)

if TYPE_CHECKING:
  from mjlab.actuator.actuator import Actuator
  from mjlab.entity.data import EntityData

BuiltinActuatorType = (
  BuiltinMotorActuator
  | BuiltinMuscleActuator
  | BuiltinPositionActuator
  | BuiltinVelocityActuator
)

# Maps (actuator_type, transmission_type) to EntityData target tensor attribute name.
_TARGET_TENSOR_MAP: dict[tuple[type[BuiltinActuatorType], TransmissionType], str] = {
  (BuiltinPositionActuator, TransmissionType.JOINT): "joint_pos_target",
  (BuiltinVelocityActuator, TransmissionType.JOINT): "joint_vel_target",
  (BuiltinMotorActuator, TransmissionType.JOINT): "joint_effort_target",
  (BuiltinPositionActuator, TransmissionType.TENDON): "tendon_len_target",
  (BuiltinVelocityActuator, TransmissionType.TENDON): "tendon_vel_target",
  (BuiltinMotorActuator, TransmissionType.TENDON): "tendon_effort_target",
  (BuiltinMotorActuator, TransmissionType.SITE): "site_effort_target",
  (BuiltinMuscleActuator, TransmissionType.JOINT): "joint_effort_target",
  (BuiltinMuscleActuator, TransmissionType.TENDON): "tendon_effort_target",
}


@dataclass(frozen=True)
class BuiltinActuatorGroup:
  """Groups builtin actuators for batch processing.

  Builtin actuators (position, velocity, motor) just pass through target values
  from entity data to control signals. This class pre-computes the mappings and
  enables direct writes without per-actuator overhead.
  """

  # Map from (BuiltinActuator type, transmission_type) to (target_ids, ctrl_ids).
  _index_groups: dict[tuple[type, TransmissionType], tuple[torch.Tensor, torch.Tensor]]

  @staticmethod
  def process(
    actuators: list[Actuator],
  ) -> tuple[BuiltinActuatorGroup, tuple[Actuator, ...]]:
    """Register builtin actuators and pre-compute their mappings.

    Args:
      actuators: List of initialized actuators to process.

    Returns:
      A tuple containing:
        - BuiltinActuatorGroup with pre-computed mappings.
        - List of custom (non-builtin) actuators.
    """

    builtin_groups: dict[tuple[type, TransmissionType], list[Actuator]] = {}
    custom_actuators: list[Actuator] = []

    # Group actuators by (type, transmission_type).
    for act in actuators:
      if isinstance(act, BuiltinActuatorType):
        key: tuple[type, TransmissionType] = (type(act), act.cfg.transmission_type)
        builtin_groups.setdefault(key, []).append(act)
      else:
        custom_actuators.append(act)

    # Return stacked indices for each (actuator_type, transmission_type) group.
    index_groups: dict[
      tuple[type, TransmissionType], tuple[torch.Tensor, torch.Tensor]
    ] = {
      key: (
        torch.cat([act.target_ids for act in acts], dim=0),
        torch.cat([act.ctrl_ids for act in acts], dim=0),
      )
      for key, acts in builtin_groups.items()
    }
    return BuiltinActuatorGroup(index_groups), tuple(custom_actuators)

  def apply_controls(self, data: EntityData) -> None:
    """Write builtin actuator controls directly to simulation data.

    Args:
      data: Entity data containing targets and control arrays.
    """
    for (actuator_type, transmission_type), (
      target_ids,
      ctrl_ids,
    ) in self._index_groups.items():
      attr_name = _TARGET_TENSOR_MAP[(actuator_type, transmission_type)]
      target_tensor = getattr(data, attr_name)
      data.write_ctrl(target_tensor[:, target_ids], ctrl_ids)
