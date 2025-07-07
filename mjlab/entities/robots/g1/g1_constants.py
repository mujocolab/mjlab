"""Unitree G1 constants."""

# fmt: off

from typing import Dict
from mjlab import MJLAB_SRC_PATH, MENAGERIE_PATH, update_assets
from mjlab.entities.robots.robot_config import RobotCfg
from mjlab.entities.robots.robot_config import KeyframeCfg, ActuatorCfg, SensorCfg

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

LEFT_FEET_GEOMS = [f"left_foot{i}_collision" for i in range(1, 4)]
RIGHT_FEET_GEOMS = [f"right_foot{i}_collision" for i in range(1, 4)]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

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
PELVIS_IMU_SITE = "imu_in_pelvis"
TORSO_IMU_SITE = "imu_in_torso"

##
# Motor specs.
##

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ ** 2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ ** 2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

ACTUATOR_5020 = ActuatorCfg(
  joint_names_expr=[
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ],
  effort_limit=25.0,
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  armature=ARMATURE_5020,
)

ACTUATOR_7520_14 = ActuatorCfg(
  joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"],
  effort_limit=88.0,
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  armature=ARMATURE_7520_14,
)

ACTUATOR_7520_22 = ActuatorCfg(
  joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
  effort_limit=139.0,
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  armature=ARMATURE_7520_22,
)

ACTUATOR_4010 = ActuatorCfg(
  joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
  effort_limit=5.0,
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  armature=ARMATURE_4010,
)

ACTUATOR_WAIST = ActuatorCfg(
  joint_names_expr=["waist_pitch_joint", "waist_roll_joint"],
  effort_limit=50.0,
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  armature=ARMATURE_5020 * 2,
)

ACTUATOR_ANKLE = ActuatorCfg(
  joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
  effort_limit=50.0,
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  armature=ARMATURE_5020 * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = KeyframeCfg(
  root_pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  use_joint_pos_for_ctrl=True,
)

KNEES_BENT_KEYFRAME = KeyframeCfg(
  root_pos=(0, 0, 0.755),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    "waist_pitch_joint": 0.073,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.0,
    "left_shoulder_roll_joint": 0.22,
    "right_shoulder_roll_joint": -0.22,
  },
  use_joint_pos_for_ctrl=True,
)

##
# Final config.
##

G1_ROBOT_CFG = RobotCfg(
  xml_path=G1_XML,
  asset_fn=get_assets,
  actuators=(
    ACTUATOR_5020,
    ACTUATOR_7520_14,
    ACTUATOR_7520_22,
    ACTUATOR_4010,
    ACTUATOR_WAIST,
    ACTUATOR_ANKLE,
  ),
  sensors={
    "gyro": SensorCfg("gyro", PELVIS_IMU_SITE, "site"),
    "local_linvel": SensorCfg("velocimeter", PELVIS_IMU_SITE, "site"),
    "upvector": SensorCfg("framezaxis", PELVIS_IMU_SITE, "site"),
  },
  soft_joint_pos_limit_factor=0.95,
  keyframes={
    "home": HOME_KEYFRAME,
    "knees_bent": KNEES_BENT_KEYFRAME,
  },
)
