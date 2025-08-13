import mujoco
from typing import Union


def is_position_actuator(actuator) -> bool:
  """Check if an actuator is a position actuator.

  This function works on both model.actuator and spec.actuator objects.
  """
  return (
    actuator.gaintype == mujoco.mjtGain.mjGAIN_FIXED
    and actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE
    and actuator.dyntype in (mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtDyn.mjDYN_FILTEREXACT)
    and actuator.gainprm[0] == -actuator.biasprm[1]
  )


def dof_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
  """Get the dimensionality of the joint in qvel."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
  """Get the dimensionality of the joint in qpos."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]
