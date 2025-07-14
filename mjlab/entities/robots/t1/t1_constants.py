"""Booster T1 constants."""

# fmt: off

from typing import Dict
from mjlab import MJLAB_SRC_PATH
from mjlab.utils.os import update_assets
from mjlab.entities.robots.actuator import ElectricActuator, reflected_inertia, rpm_to_rad
from mjlab.entities.robots.robot_config import RobotCfg
from mjlab.entities.common.config import CollisionCfg
from mjlab.entities.robots.robot_config import KeyframeCfg, ActuatorCfg, SensorCfg

##
# MJCF and assets
##

T1_XML = MJLAB_SRC_PATH / "entities" / "robots" / "t1" / "xmls" / "t1.xml"


def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  update_assets(assets, T1_XML.parent / "assets")
  return assets

##
# Actuator config.
##

ROTOR_INERTIA_ARM = 18.2e-6
GEAR_ARM = 36
ARMATURE_ARM = reflected_inertia(ROTOR_INERTIA_ARM, GEAR_ARM)

ROTOR_INERTIA_ANKLE = 25.5e-6
GEAR_ANKLE = 36
ARMATURE_ANKLE = reflected_inertia(ROTOR_INERTIA_ANKLE, GEAR_ANKLE)

ROTOR_INERTIA_HIP_RY = 51.457e-6
GEAR_HIP_RY = 25
ARMATURE_HIP_RY = reflected_inertia(ROTOR_INERTIA_HIP_RY, GEAR_HIP_RY)

ROTOR_INERTIA_HIP_P = 146.69e-6
GEAR_HIP_P = 18
ARMATURE_HIP_P = reflected_inertia(ROTOR_INERTIA_HIP_P, GEAR_HIP_P)

ROTOR_INERTIA_KNEE = 184.053e-6
GEAR_KNEE = 18
ARMATURE_KNEE = reflected_inertia(ROTOR_INERTIA_KNEE, GEAR_KNEE)

ROTOR_INERTIA_NECK = 18e-6
GEAR_NECK = 10
ARMATURE_NECK = reflected_inertia(ROTOR_INERTIA_NECK, GEAR_NECK)

ACTUATOR_ARM = ElectricActuator(
  reflected_inertia=ARMATURE_ARM,
  velocity_limit=rpm_to_rad(184.0),
  effort_limit=36.0,
)
ACTUATOR_ANKLE = ElectricActuator(
  reflected_inertia=ARMATURE_ANKLE,
  velocity_limit=rpm_to_rad(117.0),
  effort_limit=75.0,
)
ACTUATOR_HIP_RY = ElectricActuator(
  reflected_inertia=ARMATURE_HIP_RY,
  velocity_limit=rpm_to_rad(135.0),
  effort_limit=60.0,
)
ACTUATOR_HIP_P = ElectricActuator(
  reflected_inertia=ARMATURE_HIP_P,
  velocity_limit=rpm_to_rad(157.0),
  effort_limit=90.0,
)
ACTUATOR_KNEE = ElectricActuator(
  reflected_inertia=ARMATURE_KNEE,
  velocity_limit=rpm_to_rad(140.0),
  effort_limit=130.0,
)
ACTUATOR_NECK = ElectricActuator(
  reflected_inertia=ARMATURE_NECK,
  velocity_limit=rpm_to_rad(400.0),
  effort_limit=7.0,
)

NATURAL_FREQ = 5 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_ARM = ARMATURE_ARM * NATURAL_FREQ ** 2
STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ ** 2
STIFFNESS_HIP_RY = ARMATURE_HIP_RY * NATURAL_FREQ ** 2
STIFFNESS_HIP_P = ARMATURE_HIP_P * NATURAL_FREQ ** 2
STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ ** 2
STIFFNESS_NECK = ARMATURE_NECK * NATURAL_FREQ ** 2

DAMPING_ARM = 2.0 * DAMPING_RATIO * ARMATURE_ARM * NATURAL_FREQ
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ
DAMPING_HIP_RY = 2.0 * DAMPING_RATIO * ARMATURE_HIP_RY * NATURAL_FREQ
DAMPING_HIP_P = 2.0 * DAMPING_RATIO * ARMATURE_HIP_P * NATURAL_FREQ
DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ
DAMPING_NECK = 2.0 * DAMPING_RATIO * ARMATURE_NECK * NATURAL_FREQ

ACTUATOR_ARM = ActuatorCfg(
  joint_names_expr=[
    ".*_Shoulder_Pitch",
    ".*_Shoulder_Roll",
    ".*_Elbow_Pitch",
    ".*_Elbow_Yaw",
  ],
  effort_limit=ACTUATOR_ARM.effort_limit,
  armature=ACTUATOR_ARM.reflected_inertia,
  stiffness=STIFFNESS_ARM,
  damping=DAMPING_ARM,
)
ACTUATOR_HIP_ROLL_YAW = ActuatorCfg(
  joint_names_expr=[".*_Hip_Roll", ".*_Hip_Yaw", "Waist"],
  effort_limit=ACTUATOR_HIP_RY.effort_limit,
  armature=ACTUATOR_HIP_RY.reflected_inertia,
  stiffness=STIFFNESS_HIP_RY,
  damping=DAMPING_HIP_RY,
)
ACTUATOR_HIP_PITCH = ActuatorCfg(
  joint_names_expr=[".*_Hip_Pitch"],
  effort_limit=ACTUATOR_HIP_P.effort_limit,
  armature=ACTUATOR_HIP_P.reflected_inertia,
  stiffness=STIFFNESS_HIP_P,
  damping=DAMPING_HIP_P,
)
ACTUATOR_KNEE = ActuatorCfg(
  joint_names_expr=[".*_Knee_Pitch"],
  effort_limit=ACTUATOR_KNEE.effort_limit,
  armature=ACTUATOR_KNEE.reflected_inertia,
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
)
ACTUATOR_NECK = ActuatorCfg(
  joint_names_expr=["AAHead_yaw", "Head_pitch"],
  effort_limit=ACTUATOR_NECK.effort_limit,
  armature=ACTUATOR_NECK.reflected_inertia,
  stiffness=STIFFNESS_NECK,
  damping=DAMPING_NECK,
)

ACTUATOR_ANKLE = ActuatorCfg(
  joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
  effort_limit=ACTUATOR_ANKLE.effort_limit * 2,
  armature=ACTUATOR_ANKLE.reflected_inertia * 2,
  stiffness=STIFFNESS_ANKLE * 2,
  damping=DAMPING_ANKLE * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = KeyframeCfg(
  name="home",
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
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3 and custom friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],
  condim={".*_foot_collision": 3},
  priority={".*_foot_collision": 1},
  friction={".*_foot_collision": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=[".*_collision"],
  contype=0,
  conaffinity=1,
  condim={".*_foot_collision": 3},
  priority={".*_foot_collision": 1},
  friction={".*_foot_collision": (0.6,)},
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
  ),
  sensors=(
    SensorCfg("body_ang_vel", "gyro", "imu", "site"),
    SensorCfg("body_lin_vel", "velocimeter", "imu", "site"),
    SensorCfg("body_zaxis", "framezaxis", "imu", "site"),
  ),
  soft_joint_pos_limit_factor=0.95,
  keyframes=(HOME_KEYFRAME,),
  collisions=(FULL_COLLISION,)
)
