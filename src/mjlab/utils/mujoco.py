from collections.abc import Callable
from typing import Literal

import mujoco


def is_position_actuator(actuator: mujoco.MjsActuator) -> bool:
  """Check if an actuator is a position actuator.

  Matches actuators created by ``MjsActuator.set_to_position()`` or
  ``create_position_actuator()``.
  """
  return (
    actuator.gaintype == mujoco.mjtGain.mjGAIN_FIXED
    and actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE
    and actuator.dyntype in (mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtDyn.mjDYN_FILTEREXACT)
    and actuator.gainprm[0] == -actuator.biasprm[1]
  )


def is_velocity_actuator(actuator: mujoco.MjsActuator) -> bool:
  """Check if an actuator is a velocity actuator.

    Matches actuators created by ``MjsActuator.set_to_velocity()`` or
  ``create_velocity_actuator()``.
  """
  return (
    actuator.gaintype == mujoco.mjtGain.mjGAIN_FIXED
    and actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE
    and actuator.dyntype == mujoco.mjtDyn.mjDYN_NONE
    and actuator.biasprm[1] == 0
    and actuator.gainprm[0] == -actuator.biasprm[2]
  )


def is_motor_actuator(actuator: mujoco.MjsActuator) -> bool:
  """Check if an actuator is a motor actuator.

  Matches actuators created by ``MjsActuator.set_to_motor()`` or
  ``create_motor_actuator()``.
  """
  return (
    actuator.gaintype == mujoco.mjtGain.mjGAIN_FIXED
    and actuator.biastype == mujoco.mjtBias.mjBIAS_NONE
  )


def is_muscle_actuator(actuator: mujoco.MjsActuator) -> bool:
  """Check if an actuator is a muscle actuator."""
  return actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE


_ACTUATOR_PREDICATES: list[
  tuple[Literal["position", "velocity", "effort"], Callable[[mujoco.MjsActuator], bool]]
] = [
  ("effort", is_muscle_actuator),
  ("position", is_position_actuator),
  ("velocity", is_velocity_actuator),
  ("effort", is_motor_actuator),  # Broad; checked last.
]


def detect_command_field(
  actuator: mujoco.MjsActuator,
) -> Literal["position", "velocity", "effort"]:
  """Detect which ``ActuatorCmd`` field an XML actuator expects as ctrl input.

  Uses strict predicates that match the invariants set by MuJoCo's
  ``mjs_setToPosition``, ``mjs_setToVelocity``, ``mjs_setToMotor``,
  and ``mjs_setToMuscle``. Raises ``ValueError`` for unrecognized types.
  """
  for field, predicate in _ACTUATOR_PREDICATES:
    if predicate(actuator):
      return field
  raise ValueError(
    f"Cannot determine command field for XML actuator '{actuator.name}'. "
    f"Only position, velocity, motor, and muscle actuators are supported. "
    f"For general actuators, use a custom actuator class."
  )


def dof_width(joint_type: int | mujoco.mjtJoint) -> int:
  """Get the dimensionality of the joint in qvel."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: int | mujoco.mjtJoint) -> int:
  """Get the dimensionality of the joint in qpos."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]
