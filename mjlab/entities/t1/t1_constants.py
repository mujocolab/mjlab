"""Booster T1 constants."""

# fmt: off

from mjlab.entities.structs import (
  CollisionPair,
  Keyframe,
  PDActuator,
  Sensor,
)

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

FEET_GEOMS = ["left_foot", "right_foot"]

TORSO_BODY = "Trunk"
ROOT_BODY = TORSO_BODY

BODY_NAMES = []

IMU_SITE = "imu"


ACTUATOR_SPECS = (
  PDActuator(joint_name="AAHead_yaw", kp=20, kv=5, armature=0.005, torque_limit=7),
  PDActuator(joint_name="Head_pitch", kp=20, kv=5, armature=0.005, torque_limit=7),

  PDActuator(joint_name="*_Shoulder_Pitch", kp=20, kv=2, armature=0.005, torque_limit=18),
  PDActuator(joint_name="*_Shoulder_Roll", kp=20, kv=2, armature=0.005, torque_limit=18),
  PDActuator(joint_name="*_Elbow_Pitch", kp=20, kv=2, armature=0.005, torque_limit=18),
  PDActuator(joint_name="*_Elbow_Yaw", kp=20, kv=2, armature=0.005, torque_limit=18),

  PDActuator(joint_name="Waist", kp=50, kv=5, armature=0.005, torque_limit=30),

  PDActuator(joint_name="*_Hip_Pitch", kp=50, kv=5, armature=0.005, torque_limit=45),
  PDActuator(joint_name="*_Hip_Roll", kp=50, kv=5, armature=0.005, torque_limit=30),
  PDActuator(joint_name="*_Hip_Yaw", kp=50, kv=5, armature=0.005, torque_limit=30),
  PDActuator(joint_name="*_Knee_Pitch", kp=50, kv=5, armature=0.005, torque_limit=60),
  PDActuator(joint_name="*_Ankle_Pitch", kp=20, kv=2, armature=0.005, torque_limit=20),
  PDActuator(joint_name="*_Ankle_Roll", kp=20, kv=2, armature=0.005, torque_limit=15),
)

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
)


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
    Sensor("gyro", "gyro", "imu", "site"),
    Sensor("local_linvel", "velocimeter", "imu", "site"),
    Sensor("upvector", "framezaxis", "imu", "site"),
)
