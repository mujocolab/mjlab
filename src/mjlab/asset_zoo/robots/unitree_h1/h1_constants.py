"""Unitree Go1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entities.robots.robot_config import RobotCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_editor import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

H1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_h1" / "xmls" / "h1.xml"
)
assert H1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, H1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Rotor inertia.
# Ref: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/h1_description/urdf/h1.urdf#L515
# Extracted Ixx (rotation along x-axis).
ROTOR_INERTIA = 0.000111842

# Gearbox.
# HIP_GEAR_RATIO = 6
# KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

# HIP_ACTUATOR = ElectricActuator(
#   reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
#   velocity_limit=30.1,
#   effort_limit=23.7,
# )
# KNEE_ACTUATOR = ElectricActuator(
#   reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
#   velocity_limit=20.06,
#   effort_limit=35.55,
# )

# H1_LEGS_ACTUATOR_CFG = ActuatorCfg(
#   joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
#   effort_limit=300,
#   stiffness=150.0,
#   # stiffness={
#   #     ".*_hip_yaw": 150.0,
#   #     ".*_hip_roll": 150.0,
#   #     ".*_hip_pitch": 200.0,
#   #     ".*_knee": 200.0,
#   #     "torso": 200.0,
#   # },
#   damping=5.0,
#   # damping={
#   #     ".*_hip_yaw": 5.0,
#   #     ".*_hip_roll": 5.0,
#   #     ".*_hip_pitch": 5.0,
#   #     ".*_knee": 5.0,
#   #     "torso": 5.0,
#   # },
# )
# H1_FEET_ACTUATOR_CFG = ActuatorCfg(
#   joint_names_expr=[".*_ankle"],
#   effort_limit=100,
#   stiffness=20.0, #{".*_ankle": 20.0},
#   damping=4.0, #{".*_ankle": 4.0},
# )

# H1_ARMS_ACTUATOR_CFG = ActuatorCfg(
#   joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
#   effort_limit=300,
#   stiffness=40.0,
#   damping=10.0,
#   # stiffness={
#   #     ".*_shoulder_pitch": 40.0,
#   #     ".*_shoulder_roll": 40.0,
#   #     ".*_shoulder_yaw": 40.0,
#   #     ".*_elbow": 40.0,
#   # },
#   # damping={
#   #     ".*_shoulder_pitch": 10.0,
#   #     ".*_shoulder_roll": 10.0,
#   #     ".*_shoulder_yaw": 10.0,
#   #     ".*_elbow": 10.0,
#   # },
# )
# ##
# # Keyframes.
# ##


# INIT_STATE = RobotCfg.InitialStateCfg(
#   pos=(0.0, 0.0, 1.05),
#   joint_pos={
#     ".*_hip_yaw": 0.0,
#     ".*_hip_roll": 0.0,
#     ".*_hip_pitch": -0.28,  # -16 degrees
#     ".*_knee": 0.79,  # 45 degrees
#     ".*_ankle": -0.52,  # -30 degrees
#     "torso": 0.0,
#     ".*_shoulder_pitch": 0.28,
#     ".*_shoulder_roll": 0.0,
#     ".*_shoulder_yaw": 0.0,
#     ".*_elbow": 0.52,
#   },
# )


H1_LEGS_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[
    ".*_hip_yaw$",
    ".*_hip_roll$",
    ".*_hip_pitch$",
    ".*_knee_joint$",  # was .*_knee
    "torso$",  # was torso
  ],
  effort_limit=300,
  stiffness=150.0,
  damping=5.0,
)

H1_FEET_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[".*_ankle$"],
  effort_limit=100,
  stiffness=20.0,
  damping=4.0,
)

H1_ARMS_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[
    ".*_shoulder_pitch$",
    ".*_shoulder_roll$",
    ".*_shoulder_yaw$",
    ".*_elbow$",
  ],
  effort_limit=300,
  stiffness=40.0,
  damping=10.0,
)

INIT_STATE = RobotCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.05),
  joint_pos={
    ".*_hip_yaw$": 0.0,
    ".*_hip_roll$": 0.0,
    ".*_hip_pitch$": -0.28,
    ".*_knee_joint$": 0.79,  # was .*_knee
    ".*_ankle$": -0.52,
    "torso$": 0.0,  # was torso
    ".*_shoulder_pitch$": 0.28,
    ".*_shoulder_roll$": 0.0,
    ".*_shoulder_yaw$": 0.0,
    ".*_elbow$": 0.52,
  },
)


##
# Collision config.
##

_foot_regex = "^[right][left]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=[_foot_regex],
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],
  condim={_foot_regex: 3},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

H1_ROBOT_CFG = RobotCfg(
  init_state=INIT_STATE,
  actuators=(
    H1_LEGS_ACTUATOR_CFG,
    H1_FEET_ACTUATOR_CFG,
    H1_ARMS_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
)
