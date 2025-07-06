"""Unitree G1 constants."""

# fmt: off

from typing import Dict

import numpy as np
from mjlab.entities.robots.editors import (
  CollisionPair,
  Keyframe,
  Actuator,
  Joint,
  Sensor,
)
from mjlab.entities.robots.robot import RobotConfig
from mjlab import MJLAB_SRC_PATH, MENAGERIE_PATH, update_assets

##
# MJCF and assets.
##

G1_XML = MJLAB_SRC_PATH / "entities" / "robots" / "g1" / "xmls" / "g1.xml"

def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  path = MENAGERIE_PATH / "unitree_g1"
  update_assets(assets, path / "assets")
  return assets

##
# Constants.
##

NU = 29
NQ = NU + 7
NV = NQ - 1

LEFT_FOOT_SITE = "left_foot"
RIGHT_FOOT_SITE = "right_foot"

FEET_SITES = (
  LEFT_FOOT_SITE,
  RIGHT_FOOT_SITE,
)

HAND_SITES = (
  "left_palm",
  "right_palm",
)

LEFT_FEET_GEOMS = [f"left_foot{i}_collision" for i in range(1, 4)]
RIGHT_FEET_GEOMS = [f"right_foot{i}_collision" for i in range(1, 4)]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

PELVIS_BODY = "pelvis"
TORSO_BODY = "torso_link"
ROOT_BODY = PELVIS_BODY

BODY_NAMES = [
  "pelvis",  # root body
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
]

BODY_NAMES_MINUS_END_EFFECTORS = [
  "torso_link",
  "left_hip_roll_link",
  "left_knee_link",
  "right_hip_roll_link",
  "right_knee_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
]

END_EFFECTOR_NAMES = [
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
]

PELVIS_IMU_SITE = "imu_in_pelvis"
TORSO_IMU_SITE = "imu_in_torso"

##
# Actuator config.
##


ACTUATOR_CONFIG = (
  # Leg joints.
  Actuator(joint_name="*hip_pitch_joint", kp=75, kv=2, torque_limit=88),
  Actuator(joint_name="*hip_roll_joint", kp=75, kv=2, torque_limit=139),
  Actuator(joint_name="*hip_yaw_joint", kp=75, kv=2, torque_limit=88),
  Actuator(joint_name="*knee_joint", kp=75, kv=2, torque_limit=139),
  Actuator(joint_name="*ankle_pitch_joint", kp=20, kv=2, torque_limit=50),
  Actuator(joint_name="*ankle_roll_joint", kp=20, kv=2, torque_limit=50),

  # Waist joints.
  Actuator(joint_name="waist_yaw_joint", kp=75, kv=2, torque_limit=88),
  Actuator(joint_name="waist_pitch_joint", kp=75, kv=2, torque_limit=50),
  Actuator(joint_name="waist_roll_joint", kp=75, kv=2, torque_limit=50),

  # Arm joints.
  Actuator(joint_name="*shoulder_pitch_joint", kp=75, kv=2, torque_limit=25),
  Actuator(joint_name="*shoulder_roll_joint", kp=75, kv=2, torque_limit=25),
  Actuator(joint_name="*shoulder_yaw_joint", kp=75, kv=2, torque_limit=25),
  Actuator(joint_name="*elbow_joint", kp=75, kv=2, torque_limit=25),
  Actuator(joint_name="*wrist_roll_joint", kp=20, kv=2, torque_limit=25),
  Actuator(joint_name="*wrist_pitch_joint", kp=20, kv=2, torque_limit=5),
  Actuator(joint_name="*wrist_yaw_joint", kp=20, kv=2, torque_limit=5),
)

JOINT_CONFIG = (
  # Leg joints.
  Joint(joint_name="*hip_pitch_joint", armature=0.01017752004),
  Joint(joint_name="*hip_roll_joint", armature=0.025101925),
  Joint(joint_name="*hip_yaw_joint", armature=0.01017752004),
  Joint(joint_name="*knee_joint", armature=0.025101925),
  Joint(joint_name="*ankle_pitch_joint", armature=0.00721945),
  Joint(joint_name="*ankle_roll_joint", armature=0.00721945),

  # Waist joints.
  Joint(joint_name="waist_yaw_joint", armature=0.01017752004),
  Joint(joint_name="waist_pitch_joint", armature=0.00721945),
  Joint(joint_name="waist_roll_joint", armature=0.00721945),

  # Arm joints.
  Joint(joint_name="*shoulder_pitch_joint", armature=0.003609725),
  Joint(joint_name="*shoulder_roll_joint", armature=0.003609725),
  Joint(joint_name="*shoulder_yaw_joint", armature=0.003609725),
  Joint(joint_name="*elbow_joint", armature=0.003609725),
  Joint(joint_name="*wrist_roll_joint", armature=0.003609725),
  Joint(joint_name="*wrist_pitch_joint", armature=0.00425),
  Joint(joint_name="*wrist_yaw_joint", armature=0.00425),
)

##
# Collision config.
##

SELF_COLLISIONS = (
  # Foot - foot.
  CollisionPair("left_foot_box_collision", "right_foot_box_collision"),
  # Foot - shin.
  CollisionPair("left_foot_box_collision", "right_shin_collision"),
  CollisionPair("right_foot_box_collision", "left_shin_collision"),
  # Foot - linkage brace.
  CollisionPair("left_foot_box_collision", "right_linkage_brace_collision"),
  CollisionPair("right_foot_box_collision", "left_linkage_brace_collision"),
  # Hand - hip.
  CollisionPair("left_hand_collision", "left_hip_collision"),
  CollisionPair("right_hand_collision", "right_hip_collision"),
  # Hand - thigh.
  CollisionPair("left_hand_collision", "left_thigh_collision"),
  CollisionPair("right_hand_collision", "right_thigh_collision"),
  # Shin - shin.
  CollisionPair("left_shin_collision", "right_shin_collision"),
  # Torso - shoulder.
  CollisionPair("torso_collision", "left_shoulder_yaw_collision"),
  CollisionPair("torso_collision", "right_shoulder_yaw_collision"),
  # Torso - elbow.
  CollisionPair("torso_collision", "left_elbow_yaw_collision"),
  CollisionPair("torso_collision", "right_elbow_yaw_collision"),
  # Torso - wrist.
  CollisionPair("torso_collision", "left_wrist_collision"),
  CollisionPair("torso_collision", "right_wrist_collision"),
  # Torso - hands.
  CollisionPair("torso_collision", "left_hand_collision"),
  CollisionPair("torso_collision", "right_hand_collision"),
  # Thigh - thigh.
  CollisionPair("left_thigh_collision", "right_thigh_collision"),
  # Shin - thigh.
  CollisionPair("left_shin_collision", "right_thigh_collision"),
  CollisionPair("right_shin_collision", "left_thigh_collision"),
  # Hip - shin.
  CollisionPair("left_shin_collision", "right_hip_collision"),
  CollisionPair("right_shin_collision", "left_hip_collision"),
  # Hip - thigh.
  CollisionPair("left_hip_collision", "right_thigh_collision"),
  CollisionPair("right_hip_collision", "left_thigh_collision"),
  # Hand - hand.
  CollisionPair("left_hand_collision", "right_hand_collision"),
)

# Collisions between robot geoms and the floor.
FLOOR_COLLISIONS = (
  CollisionPair("head_collision", "floor", 3),
  CollisionPair("torso_collision", "floor", 3),
  CollisionPair("pelvis_collision", "floor", 3),
  CollisionPair("left_shin_collision", "floor", 3),
  CollisionPair("right_shin_collision", "floor", 3),
  CollisionPair("left_thigh_collision", "floor", 3),
  CollisionPair("right_thigh_collision", "floor", 3),
  CollisionPair("left_hip_collision", "floor", 3),
  CollisionPair("right_hip_collision", "floor", 3),
  CollisionPair("left_wrist_collision", "floor", 3),
  CollisionPair("right_wrist_collision", "floor", 3),
  CollisionPair("left_elbow_yaw_collision", "floor", 3),
  CollisionPair("right_elbow_yaw_collision", "floor", 3),
  CollisionPair("left_shoulder_yaw_collision", "floor", 3),
  CollisionPair("right_shoulder_yaw_collision", "floor", 3),
  CollisionPair("left_hand_collision", "floor", 3),
  CollisionPair("right_hand_collision", "floor", 3),
)

##
# Keyframe config.
##

_HOME_JOINT_ANGLES = [
  -0.1, 0, 0, 0.3, -0.2, 0,
  -0.1, 0, 0, 0.3, -0.2, 0,
  0, 0, 0,
  0.2, 0.2, 0, 1.28, 0, 0, 0,
  0.2, -0.2, 0, 1.28, 0, 0, 0,
]
_HOME_ROOT_POS = [0, 0, 0.783675]
_HOME_ROOT_QUAT = [1, 0, 0, 0]
HOME_KEYFRAME = Keyframe(
  name="home",
  root_pos=np.array(_HOME_ROOT_POS),
  root_quat=np.array(_HOME_ROOT_QUAT),
  joint_angles=np.array(_HOME_JOINT_ANGLES),
  ctrl=np.array(_HOME_JOINT_ANGLES),
)

_KNEE_BENT_JOINT_ANGLES = [
  -0.312, 0, 0, 0.669, -0.363, 0,
  -0.312, 0, 0, 0.669, -0.363, 0,
  0, 0, 0.073,
  0.2, 0.22, 0, 1, 0, 0, 0,
  0.2, -0.22, 0, 1, 0, 0, 0,
]
_KNEE_BENT_ROOT_POS = [0, 0, 0.755]
_KNEE_BENT_ROOT_QUAT = [1, 0, 0, 0]
KNEE_BENT_KEYFRAME = Keyframe(
  name="knees_bent",
  root_pos=np.array(_KNEE_BENT_ROOT_POS),
  root_quat=np.array(_KNEE_BENT_ROOT_QUAT),
  joint_angles=np.array(_KNEE_BENT_JOINT_ANGLES),
  ctrl=np.array(_KNEE_BENT_JOINT_ANGLES),
)

KEYFRAME_CONFIG = (
  HOME_KEYFRAME,
  KNEE_BENT_KEYFRAME,
)

##
# Sensor config.
##

SENSOR_CONFIG = (
  Sensor("gyro", "gyro", PELVIS_IMU_SITE, "site"),
  Sensor("local_linvel", "velocimeter", PELVIS_IMU_SITE, "site"),
  Sensor("upvector", "framezaxis", PELVIS_IMU_SITE, "site"),
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
