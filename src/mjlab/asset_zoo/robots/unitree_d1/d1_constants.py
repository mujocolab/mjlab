"""Unitree D1 arm constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# URDF and assets.
##

D1_URDF: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "unitree_d1"
  / "urdf"
  / "d1_description.urdf"
)
assert D1_URDF.exists()


def get_spec(*, visual_only: bool = False) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(D1_URDF))
  if visual_only:
    for geom in spec.geoms:
      geom.contype = 0
      geom.conaffinity = 0
      geom.group = 2
  return spec


##
# Actuator config.
##

D1_ARM_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Joint[1-6]",),
  stiffness=40.0,
  damping=4.0,
  effort_limit=20.0,
  armature=0.01,
)
D1_GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Joint7_.*",),
  stiffness=200.0,
  damping=10.0,
  effort_limit=30.0,
  armature=0.001,
)

D1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    D1_ARM_ACTUATOR_CFG,
    D1_GRIPPER_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)

INIT_STATE = EntityCfg.InitialStateCfg(
  joint_pos={
    ".*Joint[1-6]": 0.0,
    ".*Joint7_1": 0.0,
    ".*Joint7_2": 0.0,
  },
  joint_vel={".*": 0.0},
)


def get_d1_robot_cfg() -> EntityCfg:
  """Get a fresh D1 arm configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=D1_ARTICULATION,
  )


D1_ACTION_SCALE: dict[str, float] = {}
for a in D1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    D1_ACTION_SCALE[n] = 0.25 * e / s
