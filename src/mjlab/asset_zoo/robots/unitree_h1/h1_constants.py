"""Unitree H1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

H1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_h1" / "xmls" / "h1.xml"
)
assert H1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(H1_XML))


##
# Actuator config.
##

# Please make a PR if you know the motor specs of H1.
# As a placeholder we use the values of G1_ACTUATOR_7520_14
# that are used for G1's hip pitch, hip yaw and waist yaw.

# Known data: Actuator model M107

# Motor specs (from Unitree).

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ACTUATOR_HIP_TORSO = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=23.0,
  effort_limit=200.0,
)

ACTUATOR_KNEE = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=14.0,
  effort_limit=300.0,
)

ACTUATOR_ANKLE = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=9.0,
  effort_limit=40.0,
)

ACTUATOR_SHOULDER_PITCH_ROLL = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=9.0,
  effort_limit=40.0,
)

ACTUATOR_SHOULDER_YAW_ELBOW = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=20.0,
  effort_limit=18.0,
)

H1_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch"),
  stiffness=200.0,
  damping=5.0,
  effort_limit=ACTUATOR_HIP_TORSO.effort_limit,
  armature=ACTUATOR_HIP_TORSO.reflected_inertia,
)

H1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee",),
  stiffness=300.0,
  damping=6.0,
  effort_limit=ACTUATOR_KNEE.effort_limit,
  armature=ACTUATOR_KNEE.reflected_inertia,
)

H1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=ACTUATOR_ANKLE.effort_limit,
  armature=ACTUATOR_ANKLE.reflected_inertia,
)

H1_ACTUATOR_TORSO = BuiltinPositionActuatorCfg(
  target_names_expr=(".*torso",),
  stiffness=300.0,
  damping=6.0,
  effort_limit=ACTUATOR_HIP_TORSO.effort_limit,
  armature=ACTUATOR_HIP_TORSO.reflected_inertia,
)

H1_ACTUATOR_SHOULDER_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch", ".*_shoulder_roll"),
  stiffness=20.0,
  damping=0.5,
  effort_limit=ACTUATOR_SHOULDER_PITCH_ROLL.effort_limit,
  armature=ACTUATOR_SHOULDER_PITCH_ROLL.reflected_inertia,
)

H1_ACTUATOR_SHOULDER_YAW_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw", ".*_elbow"),
  stiffness=20.0,
  damping=0.5,
  effort_limit=ACTUATOR_SHOULDER_YAW_ELBOW.effort_limit,
  armature=ACTUATOR_SHOULDER_YAW_ELBOW.reflected_inertia,
)


##
# Keyframe config.
##

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.02),
  joint_pos={
    "left_hip_yaw": 0.0,
    "right_hip_yaw": -0.0,
    "left_hip_roll": 0.0,
    "right_hip_roll": -0.0,
    ".*_hip_pitch": -0.2,
    ".*_knee": 0.6,
    ".*_ankle": -0.4,
    ".*torso": 0.0,
    ".*_shoulder_pitch": 0.0,
    "left_shoulder_roll": 0.0,
    "right_shoulder_roll": -0.0,
    "left_shoulder_yaw": 0.0,
    "right_shoulder_yaw": -0.0,
    ".*_elbow": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-2]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-2]_collision$": 1},
  friction={r"^(left|right)_foot[1-2]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-2]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-2]_collision$": 1},
  friction={r"^(left|right)_foot[1-2]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-2]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

H1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H1_ACTUATOR_HIP,
    H1_ACTUATOR_KNEE,
    H1_ACTUATOR_ANKLE,
    H1_ACTUATOR_TORSO,
    H1_ACTUATOR_SHOULDER_PITCH_ROLL,
    H1_ACTUATOR_SHOULDER_YAW_ELBOW,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h1_robot_cfg() -> EntityCfg:
  """Get a fresh H1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H1_ARTICULATION,
  )


H1_ACTION_SCALE: dict[str, float] = {}
for a in H1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    H1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_h1_robot_cfg())

  viewer.launch(robot.spec.compile())
