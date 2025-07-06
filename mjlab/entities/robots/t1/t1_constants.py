"""Booster T1 constants."""

# fmt: off

from typing import Dict

import numpy as np
from mjlab.entities.robots.editors import (
  CollisionPair,
  Keyframe,
  Actuator,
  Sensor,
  Joint,
)
from mjlab.entities.robots.robot import RobotConfig
from mjlab import MJLAB_SRC_PATH, MENAGERIE_PATH, update_assets

##
# MJCF and assets
##

T1_XML = MJLAB_SRC_PATH / "entities" / "robots" / "t1" / "xmls" / "t1.xml"


def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  path = MENAGERIE_PATH / "booster_t1"
  update_assets(assets, path / "assets")
  return assets

##
# Constants.
##

NU = 2 + 4 * 2 + 1 + 6 * 2
NQ = NU + 7
NV = NQ - 1

# Sites.
LEFT_FOOT_SITE = "left_foot"
RIGHT_FOOT_SITE = "right_foot"
FEET_SITES = (
  LEFT_FOOT_SITE,
  RIGHT_FOOT_SITE,
)
HAND_SITES = (
  "left_hand",
  "right_hand",
)
IMU_SITE = "imu"

# Geoms.
FEET_GEOMS = ["left_foot", "right_foot"]

# Bodies.
TORSO_BODY = "Trunk"
ROOT_BODY = TORSO_BODY

##
# Actuator config.
##

ACTUATOR_CONFIG = (
  Actuator(joint_name="AAHead_yaw", kp=20, kv=5, torque_limit=7),
  Actuator(joint_name="Head_pitch", kp=20, kv=5, torque_limit=7),

  Actuator(joint_name="*_Shoulder_Pitch", kp=20, kv=2, torque_limit=18),
  Actuator(joint_name="*_Shoulder_Roll", kp=20, kv=2, torque_limit=18),
  Actuator(joint_name="*_Elbow_Pitch", kp=20, kv=2, torque_limit=18),
  Actuator(joint_name="*_Elbow_Yaw", kp=20, kv=2, torque_limit=18),

  Actuator(joint_name="Waist", kp=50, kv=5, torque_limit=30),

  Actuator(joint_name="*_Hip_Pitch", kp=50, kv=5, torque_limit=45),
  Actuator(joint_name="*_Hip_Roll", kp=50, kv=5, torque_limit=30),
  Actuator(joint_name="*_Hip_Yaw", kp=50, kv=5, torque_limit=30),
  Actuator(joint_name="*_Knee_Pitch", kp=50, kv=5, torque_limit=60),
  Actuator(joint_name="*_Ankle_Pitch", kp=20, kv=2, torque_limit=20),
  Actuator(joint_name="*_Ankle_Roll", kp=20, kv=2, torque_limit=15),
)

JOINT_CONFIG = (
  Joint(joint_name="AAHead_yaw"),
  Joint(joint_name="Head_pitch"),

  Joint(joint_name="*_Shoulder_Pitch"),
  Joint(joint_name="*_Shoulder_Roll"),
  Joint(joint_name="*_Elbow_Pitch"),
  Joint(joint_name="*_Elbow_Yaw"),

  Joint(joint_name="Waist"),

  Joint(joint_name="*_Hip_Pitch"),
  Joint(joint_name="*_Hip_Roll"),
  Joint(joint_name="*_Hip_Yaw"),
  Joint(joint_name="*_Knee_Pitch"),
  Joint(joint_name="*_Ankle_Pitch"),
  Joint(joint_name="*_Ankle_Roll"),
)

##
# Collision config.
##

SELF_COLLISIONS = (
  CollisionPair("left_hand", "left_thigh", 1),
  CollisionPair("right_hand", "right_thigh", 1),
)

# Collisions between robot geoms and the floor.
FLOOR_COLLISIONS = (
  # Trunk.
  CollisionPair("Trunk", "floor", 3),
  # Feet.
  CollisionPair("left_foot", "floor", 3),
  CollisionPair("right_foot", "floor", 3),
  # Hands.
  CollisionPair("left_hand", "floor", 3),
  CollisionPair("right_hand", "floor", 3),
  # Head.
  CollisionPair("head", "floor", 3),
  # Thighs.
  CollisionPair("left_thigh", "floor", 3),
  CollisionPair("right_thigh", "floor", 3),
  # Shin.
  CollisionPair("left_shin", "floor", 3),
  CollisionPair("right_shin", "floor", 3),
  # Knees.
  CollisionPair("left_knee", "floor", 3),
  CollisionPair("right_knee", "floor", 3),
)

##
# Keyframe config.
##

_HOME_JOINT_ANGLES = [
  0, 0,
  0, -1.4, 0, -0.4,
  0, 1.4, 0, 0.4,
  0,
  -0.2, 0, 0, 0.4, -0.2, 0,
  -0.2, 0, 0, 0.4, -0.2, 0,
]
_HOME_ROOT_POS = [0, 0, 0.665]
_HOME_ROOT_QUAT = [1, 0, 0, 0]
HOME_KEYFRAME = Keyframe(
  name="home",
  root_pos=np.array(_HOME_ROOT_POS),
  root_quat=np.array(_HOME_ROOT_QUAT),
  joint_angles=np.array(_HOME_JOINT_ANGLES),
  ctrl=np.array(_HOME_JOINT_ANGLES),
)
KEYFRAME_CONFIG = (
  HOME_KEYFRAME,
)

##
# Sensor config.
##

SENSOR_CONFIG = (
    Sensor("gyro", "gyro", IMU_SITE, "site"),
    Sensor("local_linvel", "velocimeter", IMU_SITE, "site"),
    Sensor("upvector", "framezaxis", IMU_SITE, "site"),
)

##
# Robot configs.
##

DefaultConfig = RobotConfig(
  joints=JOINT_CONFIG,
  actuators=ACTUATOR_CONFIG,
  sensors=SENSOR_CONFIG,
  keyframes=KEYFRAME_CONFIG,
)
