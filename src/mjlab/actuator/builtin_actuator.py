"""MuJoCo built-in actuators.

This module provides actuators that use MuJoCo's native actuator implementations,
created programmatically via the MjSpec API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.actuator.actuator import (
  Actuator,
  ActuatorCfg,
  ActuatorCmd,
  TransmissionType,
)
from mjlab.utils.spec import (
  apply_target_overrides,
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

  Under the hood, this creates a <position> actuator for each target and sets the
  stiffness, damping and effort limits accordingly. If armature or frictionloss are
  set, they override the corresponding joint/tendon properties from XML.
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
        viscous_damping=self.cfg.viscous_damping,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.position_target


@dataclass(kw_only=True)
class BuiltinPdActuatorCfg(ActuatorCfg):
  """Implicit-integration version of ``IdealPdActuator``.

  Both consume a position target and a velocity target with kp/kd gains.
  The difference is in how the PD is delivered to MuJoCo: ``IdealPdActuator`` computes
  the PD force in Python and feeds it to a ``<motor>`` element, which MuJoCo sees as an
  opaque external force. This actuator expresses the PD as native MuJoCo elements (a
  ``<position>`` carrying kp, a ``<velocity>`` carrying kd), so the ``implicit`` and
  ``implicitfast`` integrators include the kp/kd derivatives in their velocity update.
  That makes the actuator numerically stable at gain/timestep combinations where
  explicit Python PD would diverge, which matters when you want to run a real motor's
  stiff on-board PD gains in sim.
  """

  stiffness: float
  """Proportional gain (kp)."""
  damping: float
  """Derivative gain (kd)."""
  effort_limit: float | None = None
  """Maximum total torque applied to the joint or tendon. Enforced as a
  sum-clamp on the two PD terms via jnt_actfrcrange (JOINT) or
  tendon_actfrcrange (TENDON). None leaves the limit unset."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.transmission_type == TransmissionType.SITE:
      raise ValueError(
        "BuiltinPdActuatorCfg does not support SITE transmission. "
        "Use BuiltinMotorActuatorCfg for site transmission."
      )

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> BuiltinPdActuator:
    return BuiltinPdActuator(self, entity, target_ids, target_names)


class BuiltinPdActuator(Actuator[BuiltinPdActuatorCfg]):
  """MuJoCo native PD: paired <position> + <velocity> elements per target."""

  def __init__(
    self,
    cfg: BuiltinPdActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)

  @property
  def num_targets(self) -> int:
    """Number of targets. ``ctrl_ids`` is laid out as ``[pos..., vel...]``,
    each block of length ``num_targets``."""
    return len(self._target_ids_list)

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # Position elements first, then velocity elements, so ctrl_ids is laid out
    # as [pos_0..pos_{N-1}, vel_0..vel_{N-1}].
    for target_name in target_names:
      pos_act = create_position_actuator(
        spec,
        target_name,
        actuator_name=f"{target_name}_pd_pos",
        stiffness=self.cfg.stiffness,
        damping=0.0,  # damping lives on the <velocity> element.
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        viscous_damping=self.cfg.viscous_damping,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(pos_act)
    for target_name in target_names:
      vel_act = create_velocity_actuator(
        spec,
        target_name,
        actuator_name=f"{target_name}_pd_vel",
        damping=self.cfg.damping,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(vel_act)
    # Effort limit: sum-clamp on the joint/tendon, not on each element.
    if self.cfg.effort_limit is not None:
      lim = self.cfg.effort_limit
      for target_name in target_names:
        if self.cfg.transmission_type == TransmissionType.JOINT:
          target = spec.joint(target_name)
        else:
          target = spec.tendon(target_name)
        target.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        target.actfrcrange[:] = np.array([-lim, lim])

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return torch.cat((cmd.position_target, cmd.velocity_target), dim=1)


@dataclass(kw_only=True)
class BuiltinMotorActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in motor actuator.

  Under the hood, this creates a <motor> actuator for each target and sets its effort
  limit and gear ratio accordingly. If armature or frictionloss are set, they override
  the corresponding joint/tendon properties from XML.
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
        viscous_damping=self.cfg.viscous_damping,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    return cmd.effort_target


DcMotorInputMode = Literal["voltage", "position", "velocity"]
# Values match MuJoCo's <dcmotor input=...> enum, consumed by mjs_setToDCMotor
# and read as gainprm[8] (see mujoco/src/xml/xml_native_reader.cc dcmotorinput_map).
_DCMOTOR_INPUT_MODE: dict[DcMotorInputMode, int] = {
  "voltage": 0,
  "position": 1,
  "velocity": 2,
}


@dataclass(kw_only=True)
class BuiltinDcMotorActuatorCfg(ActuatorCfg):
  """Native MuJoCo ``<dcmotor>`` wrapper.

  Models a DC motor: torque is derived from voltage via the motor constant
  ``K`` and back-EMF, ``tau = K * (V - K * omega) / R``. The back-EMF term
  lives in ``biasprm``, so MuJoCo's ``implicit`` / ``implicitfast``
  integrators pick up its velocity derivative as effective damping.

  Three input modes select what ``ctrl`` carries:

  * "voltage": ``ctrl`` is the drive voltage. ``cmd.effort_target`` carries
    volts, not torque.
  * "position" / "velocity": an internal PID closes on the setpoint and
    the motor produces torque from its (Vmax-clamped) voltage output.

  Motor characterization: specify exactly one of

  * ``nominal=(V_n, tau_stall, omega_no_load)``: datasheet form.
  * ``motor_const=(Kt, Ke)`` + ``resistance=R``: physical form.

  ``mjs_setToDCMotor`` derives ``K`` and ``R`` (including the viscous-damping
  correction) and packs the generic ``gainprm`` / ``biasprm`` / ``dynprm``
  slots.

  Optional extensions, off by default: ``integral_gain`` / ``integral_limit``,
  ``slew_rate``, ``inductance`` / ``electrical_time_constant``, ``thermal``,
  ``lugre``, ``cogging``. ``dr.pd_gains`` randomizes only kp and kd; for DR
  over the extensions, write directly to ``actuator_gainprm`` /
  ``actuator_dynprm``.

  Different from ``DcMotorActuator``: that class is a software PD with a
  velocity-dependent torque clamp on top of a ``<motor>`` element; this one
  is the real motor model with back-EMF and an optional native PID.
  """

  mode: DcMotorInputMode = "position"
  """``ctrl`` input semantics. See class docstring."""

  nominal: tuple[float, float, float] | None = None
  """Datasheet characterization: ``(nominal_voltage, stall_torque,
  no_load_speed)``. Provide this XOR ``motor_const`` + ``resistance``."""

  motor_const: tuple[float, float] | None = None
  """Physical characterization: ``(Kt, Ke)`` (torque and back-EMF constants).
  Provide this and ``resistance`` XOR ``nominal``."""

  resistance: float | None = None
  """Terminal resistance R [Ohm]. Required when using ``motor_const``."""

  stiffness: float = 0.0
  """PID proportional gain ``kp``. Required when ``mode`` is "position" /
  "velocity". Must be 0 in "voltage" mode."""

  damping: float = 0.0
  """PID derivative gain ``kd``. Used when ``mode`` is "position" /
  "velocity". Must be 0 in "voltage" mode."""

  voltage_limit: float = 0.0
  """Max drive voltage ``Vmax``. Required when ``mode`` is "position" /
  "velocity" (clamps the PID output). In "voltage" mode it is an
  optional clamp on the user's ``ctrl``; 0 disables clamping."""

  effort_limit: float | None = None
  """Continuous torque cap [N*m]. Sets ``actuator_forcerange``. None leaves
  the per-element forcerange unset."""

  gear: float = 1.0
  """Mechanical gear ratio."""

  cogging: tuple[float, float, float] | None = None
  """Cogging torque ``(amplitude, periodicity, phase)`` in
  ``(N*m, cycles per unit length, rad)``. Models magnetic torque ripple
  from rotor-stator interaction; at joint angle ``q`` the contribution is
  ``amplitude * sin(periodicity * q + phase)``.

  Cogging is added *after* ``effort_limit`` is enforced, matching MuJoCo's
  physical model: ``effort_limit`` bounds the electromagnetic torque (the
  current limit), not the mechanical torque. Total joint torque can exceed
  ``effort_limit`` by up to ``amplitude``. None disables cogging."""

  integral_gain: float = 0.0
  """PID integral gain ``ki``. In position mode the integrator tracks
  ``ki * integral(target - q)``; in velocity mode it tracks
  ``ki * (integral(target) - q)`` (integral of velocity error w.r.t. time).
  Must be 0 in voltage mode."""

  integral_limit: float = 0.0
  """Anti-windup clamp ``Imax`` on the integrator state. ``0`` disables
  clamping (the integrator can run away)."""

  slew_rate: float = 0.0
  """Maximum rate of change of ``ctrl`` per second. Rate-limits step
  inputs; 0 disables."""

  inductance: float = 0.0
  """Winding inductance ``L`` [H]. Enables first-order electrical dynamics
  on the motor current. Internally MuJoCo uses ``te = L / R``; pass
  ``electrical_time_constant`` directly to skip the divide. 0 disables."""

  electrical_time_constant: float = 0.0
  """Alternative to ``inductance``: specify ``te`` [s] directly. Ignored
  if ``inductance > 0``. 0 disables."""

  thermal: tuple[float, float, float, float, float, float] | None = None
  """Thermal model: ``(R_thermal, C_thermal, tau_thermal, alpha, T0, T_ambient)``.
  See MuJoCo's ``<dcmotor thermal=...>`` reference for units and which
  of the first three may be underspecified. Effective resistance becomes
  ``R * (1 + alpha * (T + T_ambient - T0))``. None disables."""

  lugre: tuple[float, float, float, float, float] | None = None
  """LuGre friction: ``(sigma0, sigma1, F_Coulomb, F_Stribeck, v_Stribeck)``.
  Stick-slip friction model with bristle deflection state. Subtracted from
  joint torque after the ``effort_limit`` clamp (mechanical, like cogging).
  None disables."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.transmission_type == TransmissionType.SITE:
      raise ValueError(
        "BuiltinDcMotorActuatorCfg does not support SITE transmission. "
        "Use BuiltinMotorActuatorCfg for site transmission."
      )

    has_nominal = self.nominal is not None
    has_motor_const = self.motor_const is not None
    if has_nominal == has_motor_const:
      raise ValueError(
        "Specify exactly one of: 'nominal', or 'motor_const' (+ 'resistance')."
      )
    if has_motor_const and self.resistance is None:
      raise ValueError("'motor_const' requires 'resistance' to be set.")

    if self.mode in ("position", "velocity"):
      if self.stiffness <= 0.0:
        raise ValueError(f"{self.mode!r} mode requires stiffness > 0.")
      if self.voltage_limit <= 0.0:
        raise ValueError(f"{self.mode!r} mode requires voltage_limit > 0.")
    else:
      if self.stiffness != 0.0 or self.damping != 0.0 or self.integral_gain != 0.0:
        raise ValueError(
          "stiffness, damping, and integral_gain are unused in 'voltage' mode."
        )

    for name in (
      "integral_gain",
      "integral_limit",
      "slew_rate",
      "inductance",
      "electrical_time_constant",
    ):
      if getattr(self, name) < 0.0:
        raise ValueError(f"{name} must be non-negative.")

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> BuiltinDcMotorActuator:
    return BuiltinDcMotorActuator(self, entity, target_ids, target_names)


class BuiltinDcMotorActuator(Actuator[BuiltinDcMotorActuatorCfg]):
  """MuJoCo native ``<dcmotor>``: one actuator per target."""

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    motorconst = (
      list(self.cfg.motor_const) if self.cfg.motor_const is not None else [0.0, 0.0]
    )
    resistance = self.cfg.resistance if self.cfg.resistance is not None else 0.0
    nominal = (
      list(self.cfg.nominal) if self.cfg.nominal is not None else [0.0, 0.0, 0.0]
    )
    saturation = (
      [self.cfg.effort_limit, 0.0, 0.0]
      if self.cfg.effort_limit is not None
      else [0.0, 0.0, 0.0]
    )
    controller = [
      self.cfg.stiffness,  # kp
      self.cfg.integral_gain,  # ki
      self.cfg.damping,  # kd
      self.cfg.slew_rate,  # slewmax
      self.cfg.integral_limit,  # Imax (anti-windup)
      self.cfg.voltage_limit,  # v_max
    ]
    cogging = (
      list(self.cfg.cogging) if self.cfg.cogging is not None else [0.0, 0.0, 0.0]
    )
    inductance = [self.cfg.inductance, self.cfg.electrical_time_constant]
    thermal = list(self.cfg.thermal) if self.cfg.thermal is not None else [0.0] * 6
    lugre = list(self.cfg.lugre) if self.cfg.lugre is not None else [0.0] * 5
    input_mode = _DCMOTOR_INPUT_MODE[self.cfg.mode]

    # SITE is rejected in __post_init__, so only JOINT and TENDON remain.
    trntype = (
      mujoco.mjtTrn.mjTRN_JOINT
      if self.cfg.transmission_type == TransmissionType.JOINT
      else mujoco.mjtTrn.mjTRN_TENDON
    )

    for target_name in target_names:
      actuator = spec.add_actuator(name=target_name, target=target_name)
      actuator.trntype = trntype
      actuator.gear[0] = self.cfg.gear
      # set_to_dcmotor packs gainprm/biasprm/dynprm and sets gaintype/biastype/
      # dyntype.
      actuator.set_to_dcmotor(
        motorconst=motorconst,
        resistance=resistance,
        nominal=nominal,
        saturation=saturation,
        controller=controller,
        cogging=cogging,
        inductance=inductance,
        thermal=thermal,
        lugre=lugre,
        input_mode=input_mode,
      )

      apply_target_overrides(
        spec,
        target_name,
        self.cfg.transmission_type,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        viscous_damping=self.cfg.viscous_damping,
      )

      self._mjs_actuators.append(actuator)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    if self.cfg.mode == "position":
      return cmd.position_target
    if self.cfg.mode == "velocity":
      return cmd.velocity_target
    # voltage mode: ctrl is the drive voltage carried in effort_target.
    return cmd.effort_target


@dataclass(kw_only=True)
class BuiltinVelocityActuatorCfg(ActuatorCfg):
  """Configuration for MuJoCo built-in velocity actuator.

  Under the hood, this creates a <velocity> actuator for each target and sets the
  damping gain. If armature or frictionloss are set, they override the corresponding
  joint/tendon properties from XML.
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
        viscous_damping=self.cfg.viscous_damping,
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
