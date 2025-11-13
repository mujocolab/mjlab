"""MjSpec utils."""

import mujoco
import numpy as np


def get_non_free_joints(spec: mujoco.MjSpec) -> tuple[mujoco.MjsJoint, ...]:
  """Returns all joints except the free joint."""
  joints: list[mujoco.MjsJoint] = []
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joints.append(jnt)
  return tuple(joints)


def get_free_joint(spec: mujoco.MjSpec) -> mujoco.MjsJoint | None:
  """Returns the free joint. None if no free joint exists."""
  joint: mujoco.MjsJoint | None = None
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      joint = jnt
      break
  return joint


def disable_collision(geom: mujoco.MjsGeom) -> None:
  """Disables collision for a geom."""
  geom.contype = 0
  geom.conaffinity = 0


def is_joint_limited(jnt: mujoco.MjsJoint) -> bool:
  """Returns True if a joint is limited."""
  match jnt.limited:
    case mujoco.mjtLimited.mjLIMITED_TRUE:
      return True
    case mujoco.mjtLimited.mjLIMITED_AUTO:
      return jnt.range[0] < jnt.range[1]
    case _:
      return False


def create_motor_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  *,
  effort_limit: float,
  gear: float = 1.0,
  armature: float = 0.0,
  frictionloss: float = 0.0,
) -> mujoco.MjsActuator:
  """Create a <motor> actuator."""
  actuator = spec.add_actuator(name=joint_name, target=joint_name)

  actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_NONE

  actuator.gear[0] = gear
  # Technically redundant to set both but being explicit here.
  actuator.forcelimited = True
  actuator.forcerange[:] = np.array([-effort_limit, effort_limit])
  actuator.ctrllimited = True
  actuator.ctrlrange[:] = np.array([-effort_limit, effort_limit])

  # Joint properties.
  spec.joint(joint_name).armature = armature
  spec.joint(joint_name).frictionloss = frictionloss

  return actuator


def create_position_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  *,
  stiffness: float,
  damping: float,
  effort_limit: float | None = None,
  armature: float = 0.0,
  frictionloss: float = 0.0,
  inheritrange: float = 1.0,
) -> mujoco.MjsActuator:
  """Creates a <position> actuator."""
  actuator = spec.add_actuator(name=joint_name, target=joint_name)

  actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

  actuator.inheritrange = inheritrange
  actuator.ctrllimited = True  # Technically redundant but being explicit.
  actuator.gainprm[0] = stiffness
  actuator.biasprm[1] = -stiffness
  actuator.biasprm[2] = -damping

  if effort_limit is not None:
    # Will this throw an error with autolimits=True?
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-effort_limit, effort_limit])
  else:
    actuator.forcelimited = False

  # Joint properties.
  spec.joint(joint_name).armature = armature
  spec.joint(joint_name).frictionloss = frictionloss

  return actuator


def create_velocity_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  *,
  damping: float,
  effort_limit: float | None = None,
  armature: float = 0.0,
  frictionloss: float = 0.0,
  inheritrange: float = 1.0,
) -> mujoco.MjsActuator:
  """Creates a <velocity> actuator."""
  actuator = spec.add_actuator(name=joint_name, target=joint_name)

  actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

  actuator.inheritrange = inheritrange
  actuator.ctrllimited = True  # Technically redundant but being explicit.
  actuator.gainprm[0] = damping
  actuator.biasprm[2] = -damping

  if effort_limit is not None:
    # Will this throw an error with autolimits=True?
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-effort_limit, effort_limit])
  else:
    actuator.forcelimited = False

  # Joint properties.
  spec.joint(joint_name).armature = armature
  spec.joint(joint_name).frictionloss = frictionloss

  return actuator
