"""Unitree Go2 EDU with mounted Unitree D1 arm constants."""

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_d1.d1_constants import (
  D1_ARTICULATION,
  D1_GRIPPER_ACTUATOR_CFG,
  D1_URDF,
)
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  FULL_COLLISION as GO2_FULL_COLLISION,
)
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  GO2_ACTION_SCALE,
  GO2_ARTICULATION,
)
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  get_spec as get_go2_spec,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# Mount pose from the public go2_d1_integration xacro:
# go2_to_d1_mount origin xyz="-0.0071 0 0.087" rpy="0 0 0".
D1_MOUNT_POS = (-0.0071, 0.0, 0.087)


def _d1_visual_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(D1_URDF))
  for geom in spec.geoms:
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = 2
  return spec


def get_spec() -> mujoco.MjSpec:
  spec = get_go2_spec()
  base = spec.body("base")
  frame = base.add_frame(pos=D1_MOUNT_POS)
  spec.attach(_d1_visual_spec(), prefix="d1/", frame=frame)
  return spec


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.27),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*hip_joint": 0.0,
    ".*Joint[1-6]": 0.0,
    ".*Joint7_1": 0.0,
    ".*Joint7_2": 0.0,
  },
  joint_vel={".*": 0.0},
)

GO2_D1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    *GO2_ARTICULATION.actuators,
    *D1_ARTICULATION.actuators,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_d1_robot_cfg() -> EntityCfg:
  """Get a fresh Go2+D1 mobile manipulator configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(GO2_FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_D1_ARTICULATION,
  )


GO2_D1_ACTION_SCALE: dict[str, float] = dict(GO2_ACTION_SCALE)
for a in (D1_GRIPPER_ACTUATOR_CFG,):
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    GO2_D1_ACTION_SCALE[n] = 0.25 * e / s

for n in D1_ARTICULATION.actuators[0].target_names_expr:
  GO2_D1_ACTION_SCALE[n] = 0.05
