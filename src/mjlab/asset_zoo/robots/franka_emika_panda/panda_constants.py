"""Franka Emika Panda constants (MJX-aligned, with finger)."""

from pathlib import Path
import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


##
# MJCF and assets.
##

PANDA_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "franka_emika_panda" / "xmls" / "panda.xml"
)
assert PANDA_XML.exists(), f"Missing Panda MJCF at {PANDA_XML}"

def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, PANDA_XML.parent / "assets", meshdir)
    return assets

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec

##
# Keyframes.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(-0.041018, -0.00014, 0.049974),
  joint_pos={
    "joint1": 0,
    "joint2": 0.3,
    "joint3": 0,
    "joint4": -1.57079,
    "joint5": 0,
    "joint6": 2.0,
    "joint7": 0.714,
    "finger_joint1": 0.04,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_hand_regex = r"^(left_finger_pad|right_finger_pad|hand_capsule)_collision$"

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision_*"],
    condim={_hand_regex: 3, ".*_collision_*": 1},
    priority={_hand_regex: 1},
    friction={_hand_regex: (0.8,)},
    contype=1,
    conaffinity=1,
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=[".*_collision_*"],
    condim={_hand_regex: 3, ".*_collision_*": 1},
    priority={_hand_regex: 1},
    friction={_hand_regex: (0.8,)},
    contype=0,
    conaffinity=1,
)

HAND_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=[_hand_regex],
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.8,),
)

##
# Final config.
##

PANDA_ROBOT_CFG = EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(HAND_ONLY_COLLISION,),
    spec_fn=get_spec,
)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(PANDA_ROBOT_CFG)

  viewer.launch(robot.spec.compile())
