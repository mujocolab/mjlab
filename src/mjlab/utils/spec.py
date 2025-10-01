# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MjSpec utils."""

import mujoco


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
