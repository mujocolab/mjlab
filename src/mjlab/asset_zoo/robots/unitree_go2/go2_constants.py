"""Unitree Go2 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2_XML))


##
# Actuator config.
##

# The bundled Go2 MJCF provides torque limits but not motor rotor inertia.
# Reuse the Go1-family reflected inertia model so Go2 has the same native PD
# actuator structure as the existing quadruped task configs.
ROTOR_INERTIA = 0.000111842

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
  effort_limit=45.43,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_HIP = 2 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

GO2_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS_HIP,
  damping=DAMPING_HIP,
  effort_limit=HIP_ACTUATOR.effort_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
)
GO2_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.27),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*hip_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
  solimp=(0.015, 1, 0.022),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.8, 0.02, 0.01)},
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_HIP_ACTUATOR_CFG,
    GO2_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )


GO2_ACTION_SCALE: dict[str, float] = {}
for a in GO2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    GO2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())

  viewer.launch(robot.spec.compile())
