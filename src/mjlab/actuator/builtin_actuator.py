"""MuJoCo built-in actuators.

This module provides actuators that use MuJoCo's native actuator implementations,
created programmatically via the MjSpec API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.actuator import ActuatedType, Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import (
  create_motor_actuator,
  create_muscle_actuator,
  create_position_actuator,
  create_velocity_actuator,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class BuiltinPositionActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in position actuator.

  Under the hood, this creates a <position> actuator for each joint and sets
  the stiffness, damping and effort limits accordingly. It also modifies the
  actuated joint's properties, namely armature and frictionloss.
  """

  stiffness: float
  """PD proportional gain."""
  damping: float
  """PD derivative gain."""
  effort_limit: float | None = None
  """Maximum actuator force/torque. If None, no limit is applied."""

  actuated_type: ActuatedType = ActuatedType.JOINT
  """Type of elements being actuated: 'joint'."""

  def build(
    self, entity: Entity, actuated_ids: list[int], actuated_names: list[str]
  ) -> BuiltinPositionActuator:
    return BuiltinPositionActuator(self, entity, actuated_ids, actuated_names)


class BuiltinPositionActuator(Actuator):
  """MuJoCo built-in position actuator."""

  def __init__(
    self,
    cfg: BuiltinPositionActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(entity, joint_ids, joint_names)
    self.cfg = cfg

  def edit_spec(self, spec: mujoco.MjSpec, actuated_names: list[str]) -> None:
    # Add <position> actuator to spec, one per joint.
    for joint_name in actuated_names:
      actuator = create_position_actuator(
        spec,
        joint_name,
        stiffness=self.cfg.stiffness,
        damping=self.cfg.damping,
        effort_limit=self.cfg.effort_limit,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.position_target


@dataclass(kw_only=True)
class BuiltinMotorActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in motor actuator.

  Under the hood, this creates a <motor> actuator for each joint and sets
  its effort limit and gear ratio accordingly. It also modifies the actuated
  joint's properties, namely armature and frictionloss.
  """

  effort_limit: float
  """Maximum actuator effort."""
  gear: float = 1.0
  """Actuator gear ratio."""

  actuated_type: ActuatedType = ActuatedType.JOINT
  """Type of elements being actuated: 'joint'."""

  def build(
    self, entity: Entity, actuated_ids: list[int], actuated_names: list[str]
  ) -> BuiltinMotorActuator:
    return BuiltinMotorActuator(self, entity, actuated_ids, actuated_names)


class BuiltinMotorActuator(Actuator):
  """MuJoCo built-in motor actuator."""

  def __init__(
    self,
    cfg: BuiltinMotorActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(entity, joint_ids, joint_names)
    self.cfg = cfg

  def edit_spec(self, spec: mujoco.MjSpec, actuated_names: list[str]) -> None:
    # Add <motor> actuator to spec, one per joint.
    for joint_name in actuated_names:
      actuator = create_motor_actuator(
        spec,
        joint_name,
        effort_limit=self.cfg.effort_limit,
        gear=self.cfg.gear,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target


@dataclass(kw_only=True)
class BuiltinVelocityActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in velocity actuator.

  Under the hood, this creates a <velocity> actuator for each joint and sets
  the damping gain. It also modifies the actuated joint's properties, namely
  armature and frictionloss.
  """

  damping: float
  """Damping gain."""
  effort_limit: float | None = None
  """Maximum actuator force/torque. If None, no limit is applied."""

  actuated_type: ActuatedType = ActuatedType.JOINT
  """Type of elements being actuated: 'joint'."""

  def build(
    self, entity: Entity, actuated_ids: list[int], actuated_names: list[str]
  ) -> BuiltinVelocityActuator:
    return BuiltinVelocityActuator(self, entity, actuated_ids, actuated_names)


class BuiltinVelocityActuator(Actuator):
  """MuJoCo built-in velocity actuator."""

  def __init__(
    self,
    cfg: BuiltinVelocityActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(entity, joint_ids, joint_names)
    self.cfg = cfg

  def edit_spec(self, spec: mujoco.MjSpec, actuated_names: list[str]) -> None:
    # Add <velocity> actuator to spec, one per joint.
    for joint_name in actuated_names:
      actuator = create_velocity_actuator(
        spec,
        joint_name,
        damping=self.cfg.damping,
        effort_limit=self.cfg.effort_limit,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.velocity_target


@dataclass(kw_only=True)
class BuiltinMuscleActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in muscle actuator.

  Under the hood, this creates a <muscle> actuator for each tendon and sets
  the properties accordingly. It also modifies the actuated tendon properties.
  """

  gear: float = 1.0

  actuated_type: ActuatedType = ActuatedType.TENDON
  """Type of elements being actuated: 'tendon'."""

  def build(
    self, entity: Entity, actuated_ids: list[int], actuated_names: list[str]
  ) -> BuiltinMuscleActuator:
    return BuiltinMuscleActuator(self, entity, actuated_ids, actuated_names)


class BuiltinMuscleActuator(Actuator):
  """MuJoCo built-in muscle actuator."""

  def __init__(
    self,
    cfg: BuiltinMuscleActuatorCfg,
    entity: Entity,
    tendon_ids: list[int],
    tendon_names: list[str],
  ) -> None:
    super().__init__(entity, actuated_ids=tendon_ids, actuated_names=tendon_names)
    self.cfg = cfg

  def edit_spec(self, spec: mujoco.MjSpec, actuated_names: list[str]) -> None:
    # Add <motor> actuator to spec, one per tendon.
    for tendon_name in actuated_names:
      actuator = create_muscle_actuator(
        spec,
        tendon_name,
        gear=self.cfg.gear,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.excitation
