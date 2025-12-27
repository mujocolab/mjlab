"""MuJoCo built-in actuators.

This module provides actuators that use MuJoCo's native actuator implementations,
created programmatically via the MjSpec API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.actuator import (
  Actuator,
  ActuatorCfg,
  ActuatorCmd,
  TransmissionType,
)
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

  Under the hood, this creates a <position> actuator for each target and sets
  the stiffness, damping and effort limits accordingly. It also modifies the target's
  properties, namely armature and frictionloss.
  """

  stiffness: float
  """PD proportional gain."""
  damping: float
  """PD derivative gain."""
  effort_limit: float | None = None
  """Maximum actuator force/torque. If None, no limit is applied."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.transmission_type == TransmissionType.SITE:
      raise ValueError(
        "BuiltinPositionActuatorCfg does not support SITE transmission. "
        "Use BuiltinMotorActuatorCfg for site transmission."
      )

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> BuiltinPositionActuator:
    return BuiltinPositionActuator(self, entity, target_ids, target_names)


class BuiltinPositionActuator(Actuator[BuiltinPositionActuatorCfg]):
  """MuJoCo built-in position actuator."""

  def __init__(
    self,
    cfg: BuiltinPositionActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # Add <position> actuator to spec, one per target.
    for target_name in target_names:
      actuator = create_position_actuator(
        spec,
        target_name,
        stiffness=self.cfg.stiffness,
        damping=self.cfg.damping,
        effort_limit=self.cfg.effort_limit,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.position_target


@dataclass(kw_only=True)
class BuiltinMotorActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in motor actuator.

  Under the hood, this creates a <motor> actuator for each target and sets
  its effort limit and gear ratio accordingly. It also modifies the target's
  properties, namely armature and frictionloss.
  """

  effort_limit: float
  """Maximum actuator effort."""
  gear: float = 1.0
  """Actuator gear ratio."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> BuiltinMotorActuator:
    return BuiltinMotorActuator(self, entity, target_ids, target_names)


class BuiltinMotorActuator(Actuator[BuiltinMotorActuatorCfg]):
  """MuJoCo built-in motor actuator."""

  def __init__(
    self,
    cfg: BuiltinMotorActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # Add <motor> actuator to spec, one per target.
    for target_name in target_names:
      actuator = create_motor_actuator(
        spec,
        target_name,
        effort_limit=self.cfg.effort_limit,
        gear=self.cfg.gear,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target


@dataclass(kw_only=True)
class BuiltinVelocityActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in velocity actuator.

  Under the hood, this creates a <velocity> actuator for each target and sets the
  damping gain. It also modifies the target's properties, namely armature and
  frictionloss.
  """

  damping: float
  """Damping gain."""
  effort_limit: float | None = None
  """Maximum actuator force/torque. If None, no limit is applied."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.transmission_type == TransmissionType.SITE:
      raise ValueError(
        "BuiltinVelocityActuatorCfg does not support SITE transmission. "
        "Use BuiltinMotorActuatorCfg for site transmission."
      )

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> BuiltinVelocityActuator:
    return BuiltinVelocityActuator(self, entity, target_ids, target_names)


class BuiltinVelocityActuator(Actuator[BuiltinVelocityActuatorCfg]):
  """MuJoCo built-in velocity actuator."""

  def __init__(
    self,
    cfg: BuiltinVelocityActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # Add <velocity> actuator to spec, one per target.
    for target_name in target_names:
      actuator = create_velocity_actuator(
        spec,
        target_name,
        damping=self.cfg.damping,
        effort_limit=self.cfg.effort_limit,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.velocity_target


@dataclass(kw_only=True)
class BuiltinMuscleActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in muscle actuator."""

  length_range: tuple[float, float] = (0.0, 0.0)
  """Length range for muscle actuator."""
  gear: float = 1.0
  """Gear ratio."""
  timeconst: tuple[float, float] = (0.01, 0.04)
  """Activation and deactivation time constants."""
  tausmooth: float = 0.0
  """Smoothing time constant."""
  range: tuple[float, float] = (0.75, 1.05)
  """Operating range of normalized muscle length."""
  force: float = -1.0
  """Peak force (if -1, defaults to scale * FLV)."""
  scale: float = 200.0
  """Force scaling factor."""
  lmin: float = 0.5
  """Minimum normalized muscle length."""
  lmax: float = 1.6
  """Maximum normalized muscle length."""
  vmax: float = 1.5
  """Maximum normalized muscle velocity."""
  fpmax: float = 1.3
  """Passive force at lmax."""
  fvmax: float = 1.2
  """Active force at -vmax."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.transmission_type == TransmissionType.SITE:
      raise ValueError(
        "BuiltinMuscleActuatorCfg does not support SITE transmission. "
        "Use BuiltinMotorActuatorCfg for site transmission."
      )

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> BuiltinMuscleActuator:
    return BuiltinMuscleActuator(self, entity, target_ids, target_names)


class BuiltinMuscleActuator(Actuator[BuiltinMuscleActuatorCfg]):
  """MuJoCo built-in muscle actuator."""

  def __init__(
    self,
    cfg: BuiltinMuscleActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # Add <muscle> actuator to spec, one per target.
    for target_name in target_names:
      actuator = create_muscle_actuator(
        spec,
        target_name,
        length_range=self.cfg.length_range,
        gear=self.cfg.gear,
        timeconst=self.cfg.timeconst,
        tausmooth=self.cfg.tausmooth,
        range=self.cfg.range,
        force=self.cfg.force,
        scale=self.cfg.scale,
        lmin=self.cfg.lmin,
        lmax=self.cfg.lmax,
        vmax=self.cfg.vmax,
        fpmax=self.cfg.fpmax,
        fvmax=self.cfg.fvmax,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target
