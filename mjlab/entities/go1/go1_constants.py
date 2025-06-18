"""Unitree Go1 constants."""

# fmt: off

from mjlab.entities.structs import (
  CollisionPair,
  Keyframe,
  PDActuator,
  Sensor,
)

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
_HOME_KEYFRAME = Keyframe.initialize(
  name="home",
  root_pos=_HOME_ROOT_POS,
  root_quat=_HOME_ROOT_QUAT,
  joint_angles=_HOME_JOINT_ANGLES,
)

KEYFRAMES = (
  _HOME_KEYFRAME,
)


SENSORS = (
    Sensor("gyro", "gyro", IMU_SITE, "site"),
    Sensor("local_linvel", "velocimeter", IMU_SITE, "site"),
    Sensor("upvector", "framezaxis", IMU_SITE, "site"),
)

ACTUATOR_SPECS = (
  PDActuator(joint_name="*hip_joint", kp=35, kv=0.5, torque_limit=23.7, armature=0.005,),
  PDActuator(joint_name="*thigh_joint", kp=35, kv=0.5, torque_limit=23.7, armature=0.005),
  PDActuator(joint_name="*calf_joint", kp=35, kv=0.5, torque_limit=35.55, armature=0.005),
)

CRITICALLY_DAMPED_ACTUATOR_SPECS = (
  PDActuator(joint_name="*hip_joint", kp=33.9, kv=2.26, torque_limit=23.7, armature=0.005),
  PDActuator(joint_name="*thigh_joint", kp=29.17, kv=1.94, torque_limit=23.7, armature=0.005),
  PDActuator(joint_name="*calf_joint", kp=5.12, kv=0.34, torque_limit=35.55, armature=0.005),
)
