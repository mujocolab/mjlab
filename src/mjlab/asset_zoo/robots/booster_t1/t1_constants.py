"""Booster T1 constants."""

from pathlib import Path
import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

T1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "booster_t1" / "xmls" / "t1_23dof.xml"
)
assert T1_XML.exists(), f"T1 XML not found: {T1_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, T1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(T1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Booster T1 23-DOF joint names (extracted from XML)
T1_JOINT_NAMES = [
    # Lower body (legs) - 10 DOF
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
    # Torso - 1 DOF
    "Waist",
    # Upper body (arms) - 8 DOF
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    # Head - 2 DOF
    "AAHead_yaw",
    "Head_pitch",
]

# Actuator configuration
# T1 XML already has position actuators defined, so we use XmlPositionActuatorCfg
# Use regex patterns to match joint names (they'll be prefixed with entity name in scene)
T1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlPositionActuatorCfg(
      joint_names_expr=(
        ".*AAHead_yaw",
        ".*Head_pitch",
        ".*Left_Hip_Pitch",
        ".*Left_Hip_Roll",
        ".*Left_Hip_Yaw",
        ".*Left_Knee_Pitch",
        ".*Left_Ankle_Pitch",
        ".*Left_Ankle_Roll",
        ".*Right_Hip_Pitch",
        ".*Right_Hip_Roll",
        ".*Right_Hip_Yaw",
        ".*Right_Knee_Pitch",
        ".*Right_Ankle_Pitch",
        ".*Right_Ankle_Roll",
        ".*Waist",
        ".*Left_Shoulder_Pitch",
        ".*Left_Shoulder_Roll",
        ".*Left_Elbow_Pitch",
        ".*Left_Elbow_Yaw",
        ".*Right_Shoulder_Pitch",
        ".*Right_Shoulder_Roll",
        ".*Right_Elbow_Pitch",
        ".*Right_Elbow_Yaw",
      ),
    ),
  )
)

# Foot contact links (sphere collision geometries)
T1_FOOT_STICKING_LINKS = [
    "left_foot_sphere_1_link",
    "right_foot_sphere_1_link",
    "left_foot_sphere_2_link",
    "right_foot_sphere_2_link",
    "left_foot_sphere_3_link",
    "right_foot_sphere_3_link",
    "left_foot_sphere_4_link",
    "right_foot_sphere_4_link",
    "left_foot_sphere_5_link",
    "right_foot_sphere_5_link",
]

# Collision configuration
# Enable collisions for foot geoms with proper friction
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot.*link$",),
  condim=3,
  friction=(0.6,),
)

##
# Robot properties and initial state.
##

T1_DOF = 23
T1_HEIGHT = 1.2  # meters

# Home keyframe joint positions (from XML keyframe)
T1_HOME_QPOS = {
    # Head
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    # Left arm
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.4,
    # Right arm
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.4,
    # Waist
    "Waist": 0.0,
    # Left leg
    "Left_Hip_Pitch": -0.2,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.2,
    "Left_Ankle_Roll": 0.0,
    # Right leg
    "Right_Hip_Pitch": -0.2,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.4,
    "Right_Ankle_Pitch": -0.2,
    "Right_Ankle_Roll": 0.0,
}

# Initial state (standing position from home keyframe)
# Note: joint_pos uses regex patterns to match joints (they'll be prefixed in scene)
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.665),  # From keyframe "home" in XML
  rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion [w, x, y, z]
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
  joint_pos={
    # Head
    ".*AAHead_yaw": 0.0,
    ".*Head_pitch": 0.0,
    # Left arm
    ".*Left_Shoulder_Pitch": 0.0,
    ".*Left_Shoulder_Roll": -1.4,
    ".*Left_Elbow_Pitch": 0.0,
    ".*Left_Elbow_Yaw": -0.4,
    # Right arm
    ".*Right_Shoulder_Pitch": 0.0,
    ".*Right_Shoulder_Roll": 1.4,
    ".*Right_Elbow_Pitch": 0.0,
    ".*Right_Elbow_Yaw": 0.4,
    # Waist
    ".*Waist": 0.0,
    # Left leg
    ".*Left_Hip_Pitch": -0.2,
    ".*Left_Hip_Roll": 0.0,
    ".*Left_Hip_Yaw": 0.0,
    ".*Left_Knee_Pitch": 0.4,
    ".*Left_Ankle_Pitch": -0.2,
    ".*Left_Ankle_Roll": 0.0,
    # Right leg
    ".*Right_Hip_Pitch": -0.2,
    ".*Right_Hip_Roll": 0.0,
    ".*Right_Hip_Yaw": 0.0,
    ".*Right_Knee_Pitch": 0.4,
    ".*Right_Ankle_Pitch": -0.2,
    ".*Right_Ankle_Roll": 0.0,
  },
  joint_vel={".*": 0.0},
)


def get_t1_robot_cfg() -> EntityCfg:
  """Get a fresh T1 robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=T1_ARTICULATION,
  )


##
# Actuator force ranges (from XML).
##

T1_ACTUATOR_FRC_RANGE = {
    # Head motors
    "AAHead_yaw": (-7, 7),
    "Head_pitch": (-7, 7),
    # Arm motors
    "Left_Shoulder_Pitch": (-18, 18),
    "Left_Shoulder_Roll": (-18, 18),
    "Left_Elbow_Pitch": (-18, 18),
    "Left_Elbow_Yaw": (-18, 18),
    "Right_Shoulder_Pitch": (-18, 18),
    "Right_Shoulder_Roll": (-18, 18),
    "Right_Elbow_Pitch": (-18, 18),
    "Right_Elbow_Yaw": (-18, 18),
    # Waist motor
    "Waist": (-30, 30),
    # Leg motors
    "Left_Hip_Pitch": (-45, 45),
    "Left_Hip_Roll": (-30, 30),
    "Left_Hip_Yaw": (-30, 30),
    "Left_Knee_Pitch": (-60, 60),
    "Right_Hip_Pitch": (-45, 45),
    "Right_Hip_Roll": (-30, 30),
    "Right_Hip_Yaw": (-30, 30),
    "Right_Knee_Pitch": (-60, 60),
    # Ankle motors
    "Left_Ankle_Pitch": (-20, 20),
    "Left_Ankle_Roll": (-15, 15),
    "Right_Ankle_Pitch": (-20, 20),
    "Right_Ankle_Roll": (-15, 15),
}

##
# Joint ranges (from XML).
##

T1_JOINT_RANGES = {
    # Head
    "AAHead_yaw": (-1.57, 1.57),
    "Head_pitch": (-0.35, 1.22),
    # Left arm
    "Left_Shoulder_Pitch": (-3.31, 1.22),
    "Left_Shoulder_Roll": (-1.74, 1.57),
    "Left_Elbow_Pitch": (-2.27, 2.27),
    "Left_Elbow_Yaw": (-2.44, 0.0),
    # Right arm
    "Right_Shoulder_Pitch": (-3.31, 1.22),
    "Right_Shoulder_Roll": (-1.57, 1.74),
    "Right_Elbow_Pitch": (-2.27, 2.27),
    "Right_Elbow_Yaw": (0.0, 2.44),
    # Waist
    "Waist": (-1.57, 1.57),
    # Left leg
    "Left_Hip_Pitch": (-1.8, 1.57),
    "Left_Hip_Roll": (-0.2, 1.57),
    "Left_Hip_Yaw": (-1.0, 1.0),
    "Left_Knee_Pitch": (0.0, 2.34),
    "Left_Ankle_Pitch": (-0.87, 0.35),
    "Left_Ankle_Roll": (-0.44, 0.44),
    # Right leg
    "Right_Hip_Pitch": (-1.8, 1.57),
    "Right_Hip_Roll": (-1.57, 0.2),
    "Right_Hip_Yaw": (-1.0, 1.0),
    "Right_Knee_Pitch": (0.0, 2.34),
    "Right_Ankle_Pitch": (-0.87, 0.35),
    "Right_Ankle_Roll": (-0.44, 0.44),
}

##
# Action scale for policy training.
##

# Action scale for all joints (default 0.25 for position control)
T1_ACTION_SCALE: dict[str, float] = {name: 0.25 for name in T1_JOINT_NAMES}


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_t1_robot_cfg())
  viewer.launch(robot.spec.compile())