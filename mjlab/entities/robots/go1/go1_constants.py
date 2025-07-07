"""Unitree Go1 constants."""

# fmt: off

from typing import Dict
from mjlab import MJLAB_SRC_PATH, MENAGERIE_PATH, update_assets

from mjlab.entities.robots.robot_config import RobotCfg
from mjlab.entities.robots.robot_config import KeyframeCfg, ActuatorCfg, SensorCfg

##
# MJCF and assets.
##

GO1_XML = MJLAB_SRC_PATH / "entities" / "robots" / "go1" / "xmls" / "go1.xml"

def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  path = MENAGERIE_PATH / "unitree_go1"
  update_assets(assets, path, "*.xml")
  update_assets(assets, path / "assets")
  return assets

##
# Constants.
##

NU = 12
NQ = NU + 7
NV = NQ - 1

TORSO_BODY = "trunk"
ROOT_BODY = TORSO_BODY

FEET_GEOMS = ("FR", "FL", "RR", "RL")

FEET_SITES = ("FR", "FL", "RR", "RL")
IMU_SITE = "imu"

##
# Motor specs.
##

# Motor specs (from Unitree).
MOTOR_ROTOR_INERTIA = 0.005 / (6 ** 2)
MOTOR_VELOCITY_LIMIT = 30.1 * 6  # [rad]/[s].
MOTOR_TORQUE_LIMIT = 23.7 / 6  # [N][m].

# Actuator specs.
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5
ACTUATOR_HIP_ARMATURE = MOTOR_ROTOR_INERTIA * HIP_GEAR_RATIO ** 2
ACTUATOR_KNEE_ARMATURE = MOTOR_ROTOR_INERTIA * KNEE_GEAR_RATIO ** 2
ACTUATOR_HIP_VELOCITY_LIMIT = MOTOR_VELOCITY_LIMIT / HIP_GEAR_RATIO
ACTUATOR_KNEE_VELOCITY_LIMIT = MOTOR_VELOCITY_LIMIT / KNEE_GEAR_RATIO
ACTUATOR_HIP_TORQUE_LIMIT = MOTOR_TORQUE_LIMIT * HIP_GEAR_RATIO
ACTUATOR_KNEE_TORQUE_LIMIT = MOTOR_TORQUE_LIMIT * KNEE_GEAR_RATIO

GO1_HIP_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
  effort_limit=ACTUATOR_HIP_TORQUE_LIMIT,
  stiffness=35,
  damping=0.5,
  armature=ACTUATOR_HIP_ARMATURE,
)
GO1_KNEE_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[".*_calf_joint"],
  effort_limit=ACTUATOR_KNEE_TORQUE_LIMIT,
  stiffness=35,
  damping=0.5,
  armature=ACTUATOR_KNEE_ARMATURE,
)

##
# Keyframes.
##


GO1_HOME_KEYFRAME = KeyframeCfg(
  root_pos=(0.0, 0.0, 0.278),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  use_joint_pos_for_ctrl=True,
)

##
# Final config.
##

GO1_ROBOT_CFG = RobotCfg(
  xml_path=GO1_XML,
  asset_fn=get_assets,
  actuators=(
    GO1_HIP_ACTUATOR_CFG,
    GO1_KNEE_ACTUATOR_CFG,
  ),
  sensors={
    "gyro": SensorCfg("gyro", IMU_SITE, "site"),
    "local_linvel": SensorCfg("velocimeter", IMU_SITE, "site"),
    "upvector": SensorCfg("framezaxis", IMU_SITE, "site"),
  },
  soft_joint_pos_limit_factor=0.9,
  keyframes={
    "home": GO1_HOME_KEYFRAME,
  },
)
