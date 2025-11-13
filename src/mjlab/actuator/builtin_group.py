from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuator,
  BuiltinPositionActuator,
  BuiltinVelocityActuator,
)

if TYPE_CHECKING:
  from mjlab.actuator.actuator import Actuator
  from mjlab.entity.data import EntityData

BUILTIN_TYPES = {BuiltinMotorActuator, BuiltinPositionActuator, BuiltinVelocityActuator}


@dataclass(frozen=True)
class BuiltinActuatorGroup:
  """Groups builtin actuators for batch processing.

  Builtin actuators (position, velocity, motor) just pass through target values
  from entity data to control signals. This class pre-computes the mappings and
  enables direct writes without per-actuator overhead.
  """

  # Map from BuiltinActuator type to (joint_ids, ctrl_ids).
  _index_groups: dict[type, tuple[torch.Tensor, torch.Tensor]]

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

    builtin_groups: dict[type, list[Actuator]] = {}
    custom_actuators: list[Actuator] = []

    # Group actuators by type.
    for act in actuators:
      if type(act) in BUILTIN_TYPES:
        builtin_groups.setdefault(type(act), []).append(act)
      else:
        custom_actuators.append(act)

    # Return stacked indices for each builtin actuator type.
    index_groups = {
      k: (
        torch.cat([act.joint_ids for act in v], dim=0),
        torch.cat([act.ctrl_ids for act in v], dim=0),
      )
      for k, v in builtin_groups.items()
    }
    return BuiltinActuatorGroup(index_groups), tuple(custom_actuators)

  def apply_controls(self, data: EntityData) -> None:
    """Write builtin actuator controls directly to simulation data.

    Args:
      data: Entity data containing targets and control arrays.
    """
    target_tensor_map = {
      BuiltinPositionActuator: data.joint_pos_target,
      BuiltinVelocityActuator: data.joint_vel_target,
      BuiltinMotorActuator: data.joint_effort_target,
    }
    for actuator_type, index_group in self._index_groups.items():
      joint_ids, ctrl_ids = index_group
      data.write_ctrl(target_tensor_map[actuator_type][:, joint_ids], ctrl_ids)
