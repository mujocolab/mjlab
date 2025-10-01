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

import mujoco


def is_position_actuator(actuator: mujoco.MjsActuator) -> bool:
  """Check if an actuator is a position actuator.

  This function works on both model.actuator and spec.actuator objects.
  """
  return (
    actuator.gaintype == mujoco.mjtGain.mjGAIN_FIXED
    and actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE
    and actuator.dyntype in (mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtDyn.mjDYN_FILTEREXACT)
    and actuator.gainprm[0] == -actuator.biasprm[1]
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
