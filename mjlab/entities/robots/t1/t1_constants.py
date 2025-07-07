"""Booster T1 constants."""

# fmt: off

from typing import Dict
from mjlab import MJLAB_SRC_PATH, MENAGERIE_PATH, update_assets
from mjlab.entities.robots.robot_config import RobotCfg
from mjlab.entities.robots.robot_config import KeyframeCfg, ActuatorCfg, SensorCfg

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

# Bodies.
TORSO_BODY = "Trunk"
ROOT_BODY = TORSO_BODY

# Geoms.
FEET_GEOMS = ["left_foot", "right_foot"]

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

##
# Actuator config.
##

ARM_ROTOR_INERTIA = 18.2 * 1e-6
ARM_GEAR_RATIO = 36
ARM_ARMATURE = ARM_ROTOR_INERTIA * ARM_GEAR_RATIO ** 2

ANKLE_ROTOR_INERTIA = 25.5 * 1e-6
ANKLE_GEAR_RATIO = 36
ANKLE_ARMATURE = ANKLE_ROTOR_INERTIA * ANKLE_GEAR_RATIO ** 2

HIP_ROLL_YAW_ROTOR_INERTIA = 51.457 * 1e-6
HIP_ROLL_YAW_GEAR_RATIO = 25
HIP_ROLL_YAW_ARMATURE = HIP_ROLL_YAW_ROTOR_INERTIA * HIP_ROLL_YAW_GEAR_RATIO ** 2

HIP_PITCH_ROTOR_INERTIA = 146.69 * 1e-6
HIP_PITCH_GEAR_RATIO = 18
HIP_PITCH_ARMATURE = HIP_PITCH_ROTOR_INERTIA * HIP_PITCH_GEAR_RATIO ** 2

KNEE_ROTOR_INERTIA = 184.053 * 1e-6
KNEE_GEAR_RATIO = 18
KNEE_ARMATURE = KNEE_ROTOR_INERTIA * KNEE_GEAR_RATIO ** 2

NECK_ROTOR_INERTIA = 18.0 * 1e-6
NECK_GEAR_RATIO = 10
NECK_ARMATURE = NECK_ROTOR_INERTIA * NECK_GEAR_RATIO ** 2

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_ARM = ARM_ARMATURE * NATURAL_FREQ ** 2
STIFFNESS_ANKLE = ANKLE_ARMATURE * NATURAL_FREQ ** 2
STIFFNESS_HIP_ROLL_YAW = HIP_ROLL_YAW_ARMATURE * NATURAL_FREQ ** 2
STIFFNESS_HIP_PITCH = HIP_PITCH_ARMATURE * NATURAL_FREQ ** 2
STIFFNESS_KNEE = KNEE_ARMATURE * NATURAL_FREQ ** 2
STIFFNESS_NECK = NECK_ARMATURE * NATURAL_FREQ ** 2

DAMPING_ARM = 2.0 * DAMPING_RATIO * ARM_ARMATURE * NATURAL_FREQ
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ANKLE_ARMATURE * NATURAL_FREQ
DAMPING_HIP_ROLL_YAW = 2.0 * DAMPING_RATIO * HIP_ROLL_YAW_ARMATURE * NATURAL_FREQ
DAMPING_HIP_PITCH = 2.0 * DAMPING_RATIO * HIP_PITCH_ARMATURE * NATURAL_FREQ
DAMPING_KNEE = 2.0 * DAMPING_RATIO * KNEE_ARMATURE * NATURAL_FREQ
DAMPING_NECK = 2.0 * DAMPING_RATIO * NECK_ARMATURE * NATURAL_FREQ

ACTUATOR_ARM = ActuatorCfg(
  joint_names_expr=[
    ".*_Shoulder_Pitch",
    ".*_Shoulder_Roll",
    ".*_Elbow_Pitch",
    ".*_Elbow_Yaw",
  ],
  effort_limit=36.0,
  stiffness=STIFFNESS_ARM,
  damping=DAMPING_ARM,
  armature=ARM_ARMATURE,
)

ACTUATOR_ANKLE = ActuatorCfg(
  joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
  effort_limit=75.0,
  stiffness=STIFFNESS_ANKLE,
  damping=DAMPING_ANKLE,
  armature=ANKLE_ARMATURE,
)

ACTUATOR_HIP_ROLL_YAW = ActuatorCfg(
  joint_names_expr=[".*_Hip_Roll", ".*_Hip_Yaw",],
  effort_limit=60.0,
  stiffness=STIFFNESS_HIP_ROLL_YAW,
  damping=DAMPING_HIP_ROLL_YAW,
  armature=HIP_ROLL_YAW_ARMATURE,
)

ACTUATOR_HIP_PITCH = ActuatorCfg(
  joint_names_expr=[".*_Hip_Pitch"],
  effort_limit=90.0,
  stiffness=STIFFNESS_HIP_PITCH,
  damping=DAMPING_HIP_PITCH,
  armature=HIP_PITCH_ARMATURE,
)

ACTUATOR_KNEE = ActuatorCfg(
  joint_names_expr=[".*_Knee_Pitch"],
  effort_limit=130.0,
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  armature=KNEE_ARMATURE,
)

ACTUATOR_NECK = ActuatorCfg(
  joint_names_expr=["AAHead_yaw", "Head_pitch"],
  effort_limit=7.0,
  stiffness=STIFFNESS_NECK,
  damping=DAMPING_NECK,
  armature=NECK_ARMATURE,
)

# FIX.
ACTUATOR_WAIST = ActuatorCfg(
  joint_names_expr=["Waist"],
  effort_limit=90.0,
  stiffness=STIFFNESS_HIP_PITCH,
  damping=DAMPING_HIP_PITCH,
  armature=HIP_PITCH_ARMATURE,
)

##
# Keyframe config.
##

HOME_KEYFRAME = KeyframeCfg(
  root_pos=(0, 0, 0.665),
  joint_pos={
    "Left_Shoulder_Roll": -1.4,
    "Right_Shoulder_Roll": 1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Elbow_Yaw": 0.4,
    ".*_Hip_Pitch": -0.2,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.2,
  },
  use_joint_pos_for_ctrl=True,
)

##
# Final config.
##

T1_ROBOT_CFG = RobotCfg(
  xml_path=T1_XML,
  asset_fn=get_assets,
  actuators=(
    ACTUATOR_ARM,
    ACTUATOR_ANKLE,
    ACTUATOR_HIP_ROLL_YAW,
    ACTUATOR_HIP_PITCH,
    ACTUATOR_KNEE,
    ACTUATOR_NECK,
    ACTUATOR_WAIST,
  ),
  sensors={
    "gyro": SensorCfg("gyro", IMU_SITE, "site"),
    "local_linvel": SensorCfg("velocimeter", IMU_SITE, "site"),
    "upvector": SensorCfg("framezaxis", IMU_SITE, "site"),
  },
  soft_joint_pos_limit_factor=0.95,
  keyframes={
    "home": HOME_KEYFRAME,
  },
)
