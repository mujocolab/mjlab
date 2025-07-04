"""Unitree Go1 constants."""

# fmt: off

import numpy as np
from mjlab.entities.robot_config import (
  Keyframe,
  Actuator,
  Joint,
  Sensor,
)
from typing import Dict
from mjlab import MJLAB_SRC_PATH, MENAGERIE_PATH, update_assets

GO1_XML = MJLAB_SRC_PATH / "entities" / "go1" / "xmls" / "go1.xml"

def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  path = MENAGERIE_PATH / "unitree_go1"
  update_assets(assets, path, "*.xml")
  update_assets(assets, path / "assets")
  return assets

NU = 12
NQ = NU + 7
NV = NQ - 1

TORSO_BODY = "trunk"
ROOT_BODY = TORSO_BODY

FEET_SITES = ("FR", "FL", "RR", "RL")
FEET_GEOMS = ("FR", "FL", "RR", "RL")

IMU_SITE = "imu"

_HOME_JOINT_ANGLES = [
  0.1, 0.9, -1.8,
  -0.1, 0.9, -1.8,
  0.1, 0.9, -1.8,
  -0.1, 0.9, -1.8,
]
_HOME_ROOT_POS = [0, 0, 0.278]
_HOME_ROOT_QUAT = [1, 0, 0, 0]
_HOME_KEYFRAME = Keyframe(
  name="home",
  root_pos=np.array(_HOME_ROOT_POS),
  root_quat=np.array(_HOME_ROOT_QUAT),
  joint_angles=np.array(_HOME_JOINT_ANGLES),
  ctrl=np.array(_HOME_JOINT_ANGLES),
)

KEYFRAME_CONFIG = (
  _HOME_KEYFRAME,
)


SENSOR_CONFIG = (
    Sensor("gyro", "gyro", IMU_SITE, "site"),
    Sensor("local_linvel", "velocimeter", IMU_SITE, "site"),
    Sensor("upvector", "framezaxis", IMU_SITE, "site"),
)

# Motor specs (from Unitree).
MOTOR_ROTOR_INERTIA = 0.005 / (6 ** 2)
MOTOR_VELOCITY_LIMIT = 30.1 * 6  # [rad]/[s].
MOTOR_TORQUE_LIMIT = 23.7 / 6  # [N][m].

# Actuator specs.
## Gear ratio.
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5
## Reflected inertia.
ACTUATOR_HIP_ARMATURE = MOTOR_ROTOR_INERTIA * HIP_GEAR_RATIO ** 2
ACTUATOR_KNEE_ARMATURE = MOTOR_ROTOR_INERTIA * KNEE_GEAR_RATIO ** 2
## Velocity limits.
ACTUATOR_HIP_VELOCITY_LIMIT = MOTOR_VELOCITY_LIMIT / HIP_GEAR_RATIO
ACTUATOR_KNEE_VELOCITY_LIMIT = MOTOR_VELOCITY_LIMIT / KNEE_GEAR_RATIO
## Torque limits.
ACTUATOR_HIP_TORQUE_LIMIT = MOTOR_TORQUE_LIMIT * HIP_GEAR_RATIO
ACTUATOR_KNEE_TORQUE_LIMIT = MOTOR_TORQUE_LIMIT * KNEE_GEAR_RATIO

ACTUATOR_CONFIG = (
  Actuator(joint_name="*hip_joint", kp=35, kv=0.5, torque_limit=ACTUATOR_HIP_TORQUE_LIMIT),
  Actuator(joint_name="*thigh_joint", kp=35, kv=0.5, torque_limit=ACTUATOR_HIP_TORQUE_LIMIT),
  Actuator(joint_name="*calf_joint", kp=35, kv=0.5, torque_limit=ACTUATOR_KNEE_TORQUE_LIMIT),
)

CRITICALLY_DAMPED_ACTUATOR_CONFIG = (
  Actuator(joint_name="*hip_joint", kp=33.9, kv=2.26, torque_limit=ACTUATOR_HIP_TORQUE_LIMIT),
  Actuator(joint_name="*thigh_joint", kp=29.17, kv=1.94, torque_limit=ACTUATOR_HIP_TORQUE_LIMIT),
  Actuator(joint_name="*calf_joint", kp=5.12, kv=0.34, torque_limit=ACTUATOR_KNEE_TORQUE_LIMIT),
)

JOINT_CONFIG = (
  Joint(joint_name="*hip_joint", armature=ACTUATOR_HIP_ARMATURE),
  Joint(joint_name="*thigh_joint", armature=ACTUATOR_HIP_ARMATURE),
  Joint(joint_name="*calf_joint", armature=ACTUATOR_KNEE_ARMATURE),
)
