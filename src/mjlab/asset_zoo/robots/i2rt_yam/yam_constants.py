"""YAM constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflect_rotary_to_linear,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

YAM_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "i2rt_yam" / "xmls" / "yam.xml"
)
assert YAM_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, YAM_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(YAM_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

EFFECTIVE_INERTIAS = {
  "joint1": 0.123153,
  "joint2": 0.277411,
  "joint3": 0.232763,
  "joint4": 0.030154,
  "joint5": 0.009126,
  "joint6": 0.002868,
  "left_finger": 2.781624,
}

ARMATURE_DM_4340 = 0.032
ARMATURE_DM_4310 = 0.0018

# Reference: https://github.com/i2rt-robotics/i2rt/blob/cbe48976b44aae45af856c62545be00ea2feed11/i2rt/motor_drivers/utils.py#L159-L169
DM_4340 = ElectricActuator(
  reflected_inertia=ARMATURE_DM_4340,
  velocity_limit=10.0,
  effort_limit=28.0,
)
# Reference: https://github.com/i2rt-robotics/i2rt/blob/cbe48976b44aae45af856c62545be00ea2feed11/i2rt/motor_drivers/utils.py#L139-L149
DM_4310 = ElectricActuator(
  reflected_inertia=ARMATURE_DM_4310,
  velocity_limit=30.0,
  effort_limit=10.0,
)

NATURAL_FREQ = 2 * 2.0 * 3.1415926535  # 2Hz
DAMPING_RATIO = 2.0

# Calculate per-joint gains using effective inertia.
STIFFNESS_JOINT1 = EFFECTIVE_INERTIAS["joint1"] * NATURAL_FREQ**2
STIFFNESS_JOINT2 = EFFECTIVE_INERTIAS["joint2"] * NATURAL_FREQ**2
STIFFNESS_JOINT3 = EFFECTIVE_INERTIAS["joint3"] * NATURAL_FREQ**2
STIFFNESS_JOINT4 = EFFECTIVE_INERTIAS["joint4"] * NATURAL_FREQ**2
STIFFNESS_JOINT5 = EFFECTIVE_INERTIAS["joint5"] * NATURAL_FREQ**2
STIFFNESS_JOINT6 = EFFECTIVE_INERTIAS["joint6"] * NATURAL_FREQ**2

DAMPING_JOINT1 = 2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS["joint1"] * NATURAL_FREQ
DAMPING_JOINT2 = 2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS["joint2"] * NATURAL_FREQ
DAMPING_JOINT3 = 2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS["joint3"] * NATURAL_FREQ
DAMPING_JOINT4 = 2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS["joint4"] * NATURAL_FREQ
DAMPING_JOINT5 = 2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS["joint5"] * NATURAL_FREQ
DAMPING_JOINT6 = 2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS["joint6"] * NATURAL_FREQ

ACTUATOR_JOINT1 = BuiltinPositionActuatorCfg(
  target_names_expr=("joint1",),
  stiffness=STIFFNESS_JOINT1,
  damping=DAMPING_JOINT1,
  effort_limit=DM_4340.effort_limit,
  armature=DM_4340.reflected_inertia,
)
ACTUATOR_JOINT2 = BuiltinPositionActuatorCfg(
  target_names_expr=("joint2",),
  stiffness=STIFFNESS_JOINT2,
  damping=DAMPING_JOINT2,
  effort_limit=DM_4340.effort_limit,
  armature=DM_4340.reflected_inertia,
)
ACTUATOR_JOINT3 = BuiltinPositionActuatorCfg(
  target_names_expr=("joint3",),
  stiffness=STIFFNESS_JOINT3,
  damping=DAMPING_JOINT3,
  effort_limit=DM_4340.effort_limit,
  armature=DM_4340.reflected_inertia,
)
ACTUATOR_JOINT4 = BuiltinPositionActuatorCfg(
  target_names_expr=("joint4",),
  stiffness=STIFFNESS_JOINT4,
  damping=DAMPING_JOINT4,
  effort_limit=DM_4310.effort_limit,
  armature=DM_4310.reflected_inertia,
)
ACTUATOR_JOINT5 = BuiltinPositionActuatorCfg(
  target_names_expr=("joint5",),
  stiffness=STIFFNESS_JOINT5,
  damping=DAMPING_JOINT5,
  effort_limit=DM_4310.effort_limit,
  armature=DM_4310.reflected_inertia,
)
ACTUATOR_JOINT6 = BuiltinPositionActuatorCfg(
  target_names_expr=("joint6",),
  stiffness=STIFFNESS_JOINT6,
  damping=DAMPING_JOINT6,
  effort_limit=DM_4310.effort_limit,
  armature=DM_4310.reflected_inertia,
)

##
# Gripper transmission parameters.
##

# Reference: https://github.com/i2rt-robotics/i2rt/blob/cbe48976b44aae45af856c62545be00ea2feed11/i2rt/robots/utils.py#L106-L118

# Crank gripper: DM 4310 drives a crank arm that converts rotation to linear motion.
# The mechanism geometry (8° to 170°) provides 71mm stroke, but we operate over
# 10° to 165° (2.7 rad motor range) for safety, giving ~70mm usable stroke.
# Note: Transmission ratio varies with position as r(θ) = r_crank*sin(θ); we use
# an effective average (dx/dθ) for simulation.
GRIPPER_MOTOR_STROKE_CRANK = 2.7  # [rad]: operational motor range (from limits)
GRIPPER_LINEAR_STROKE_CRANK = 0.071  # [m]: design stroke (full mechanism range)
GRIPPER_TRANSMISSION_RATIO_CRANK = (
  GRIPPER_LINEAR_STROKE_CRANK / GRIPPER_MOTOR_STROKE_CRANK
)

# Reflect motor properties to linear gripper joint.
(
  ARMATURE_DM_4310_LINEAR_CRANK,
  VELOCITY_LIMIT_DM_4310_LINEAR_CRANK,
  EFFORT_LIMIT_DM_4310_LINEAR_CRANK,
) = reflect_rotary_to_linear(
  armature_rotary=ARMATURE_DM_4310,
  velocity_limit_rotary=DM_4310.velocity_limit,
  effort_limit_rotary=DM_4310.effort_limit,
  transmission_ratio=GRIPPER_TRANSMISSION_RATIO_CRANK,
)

# PD controller gains using effective inertia.
NATURAL_FREQ_GRIPPER = 1.0 * 2.0 * 3.1415926535  # 1Hz
STIFFNESS_GRIPPER = EFFECTIVE_INERTIAS["left_finger"] * NATURAL_FREQ_GRIPPER**2
DAMPING_GRIPPER = (
  2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS["left_finger"] * NATURAL_FREQ_GRIPPER
)

# Artificially limit gripper force for sim stability (must also be done on hardware).
EFFORT_LIMIT_DM_4310_LINEAR_CRANK_SAFE = EFFORT_LIMIT_DM_4310_LINEAR_CRANK * 0.1

# Only actuate left_finger; right_finger is coupled via equality constraint.
ACTUATOR_DM_4310_LINEAR_CRANK = BuiltinPositionActuatorCfg(
  target_names_expr=("left_finger",),
  stiffness=STIFFNESS_GRIPPER,
  damping=DAMPING_GRIPPER,
  effort_limit=EFFORT_LIMIT_DM_4310_LINEAR_CRANK_SAFE,
  armature=ARMATURE_DM_4310_LINEAR_CRANK,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.01),
  joint_pos={
    "joint2": 1.047,
    "joint3": 1.05,
    "joint4": -0.9,
    "left_finger": 0.0375 / 2,
    "right_finger": -0.0375 / 2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    "[lr]f_down(6|7|8|9|10|11)_collision": 6,
    ".*_collision": 3,
  },
  friction={
    "[lr]f_down(6|7|8|9|10|11)_collision": (1, 5e-3, 5e-4),
    ".*_collision": (0.6,),
  },
  solref={
    "[lr]f_down(6|7|8|9|10|11)_collision": (0.01, 1),
  },
  priority={
    "[lr]f_down(6|7|8|9|10|11)_collision": 1,
  },
)

GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype={
    "(link6|[lr]f)_.*_collision": 1,
    ".*_collision": 0,
  },
  conaffinity={
    "(link6|[lr]f)_.*_collision": 1,
    ".*_collision": 0,
  },
  condim={
    "[lr]f_down(6|7|8|9|10|11)_collision": 6,
    ".*_collision": 3,
  },
  friction={
    "[lr]f_down(6|7|8|9|10|11)_collision": (1, 5e-3, 5e-4),
    ".*_collision": (0.6,),
  },
  solref={
    "[lr]f_down(6|7|8|9|10|11)_collision": (0.01, 1),
  },
  priority={
    "[lr]f_down(6|7|8|9|10|11)_collision": 1,
  },
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ACTUATOR_JOINT1,
    ACTUATOR_JOINT2,
    ACTUATOR_JOINT3,
    ACTUATOR_JOINT4,
    ACTUATOR_JOINT5,
    ACTUATOR_JOINT6,
    ACTUATOR_DM_4310_LINEAR_CRANK,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_yam_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


YAM_ACTION_SCALE: dict[str, float] = {}
for a in ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    YAM_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_yam_robot_cfg())

  viewer.launch(robot.spec.compile())
