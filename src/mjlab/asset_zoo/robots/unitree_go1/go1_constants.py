"""Unitree Go1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_editor import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

GO1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go1" / "xmls" / "go1.xml"
)
assert GO1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


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
  stiffness=35.0,
  damping=0.5,
  armature=HIP_ACTUATOR.reflected_inertia,
)
GO1_KNEE_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[".*_calf_joint"],
  effort_limit=KNEE_ACTUATOR.effort_limit,
  stiffness=35.0,
  damping=0.5,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.278),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

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

GO1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO1_HIP_ACTUATOR_CFG,
    GO1_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)

GO1_ROBOT_CFG = EntityCfg(
  init_state=INIT_STATE,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=GO1_ARTICULATION,
)
