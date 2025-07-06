"""MjSpec utils."""

from typing import Tuple

import mujoco


def get_non_root_joints(spec: mujoco.MjSpec) -> Tuple[mujoco.MjsJoint]:
  """Returns all joints except the root joint."""
  joints = []
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joints.append(jnt)
  return tuple(joints)


def get_root_joint(spec: mujoco.MjSpec) -> mujoco.MjsJoint | None:
  """Returns the root joint. None if no root joint exists."""
  joint = None
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      joint = jnt
      break
  return joint
