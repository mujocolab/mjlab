"""Unitree Go1 constants."""

# fmt: off

from typing import Dict
from mjlab import MJLAB_SRC_PATH
from mjlab.utils.os import update_assets

from mjlab.entities.robots.actuator import ElectricActuator, reflected_inertia
from mjlab.entities.robots.robot_config import RobotCfg
from mjlab.entities.robots.robot_config import InitialStateCfg, ActuatorCfg, SensorCfg, CollisionCfg

##
# MJCF and assets.
##

GO1_XML = MJLAB_SRC_PATH / "entities" / "robots" / "go1" / "xmls" / "go1.xml"

def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  update_assets(assets, GO1_XML.parent / "assets")
  return assets

##
# Actuator config.
##

# Rotor inertia.
# Ref: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go1_description/urdf/go1.urdf#L515
# Extracted Ixx (rotation along x-axis).
ROTOR_INERTIA = 0.000111842

# Gearbox.
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=30.1,
  effort_limit=23.7,
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=20.06,
  effort_limit=35.55,
)

GO1_HIP_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
  effort_limit=HIP_ACTUATOR.effort_limit,
  stiffness=35,
  damping=0.5,
  armature=HIP_ACTUATOR.reflected_inertia,
)
GO1_KNEE_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[".*_calf_joint"],
  effort_limit=KNEE_ACTUATOR.effort_limit,
  stiffness=35,
  damping=0.5,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##


INIT_STATE = InitialStateCfg(
  pos=(0.0, 0.0, 0.278),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
)

##
# Collision config.
##

# This disables all collisions except the feet.
# Furthermore feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=[".*_foot_collision"],
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3 and custom friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],
  condim={".*_foot_collision": 3},
  priority={".*_foot_collision": 1},
  friction={".*_foot_collision": (0.6,)},
  solimp={".*_foot_collision": (0.9, 0.95, 0.023)}
)

##
# Final config.
##

GO1_ROBOT_CFG = RobotCfg(
  xml_path=GO1_XML,
  asset_fn=get_assets,
  init_state=INIT_STATE,
  actuators=(
    GO1_HIP_ACTUATOR_CFG,
    GO1_KNEE_ACTUATOR_CFG,
  ),
  sensors=(
    SensorCfg("body_ang_vel", "gyro", "imu", "site"),
    SensorCfg("body_lin_vel", "velocimeter", "imu", "site"),
    SensorCfg("body_zaxis", "framezaxis", "imu", "site"),
  ),
  soft_joint_pos_limit_factor=0.95,
  collisions=(FEET_ONLY_COLLISION,),
)
