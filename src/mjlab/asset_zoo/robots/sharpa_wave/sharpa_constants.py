"""Sharpa Wave hand constants."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Literal

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import ObjRef, TactileSensorCfg
from mjlab.utils.spec_config import CollisionCfg

Side = Literal["left", "right"]

##
# MJCF and assets.
##

SHARPA_XML_DIR = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "sharpa_wave" / "xmls"
assert (SHARPA_XML_DIR / "left_hand.xml").exists()
assert (SHARPA_XML_DIR / "right_hand.xml").exists()


def get_spec(side: Side) -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(SHARPA_XML_DIR / f"{side}_hand.xml"))


##
# Actuator config.
##

# Joint and PD parameters are ported from MuJoCo Menagerie. The damping gains are
# the kv values MuJoCo derives from the Menagerie dampratio of 0.9.


@dataclass(frozen=True)
class _JointClass:
  armature: float
  frictionloss: float
  damping: float
  effort_limit: float


_CMC = _JointClass(
  armature=0.00451, frictionloss=0.132, damping=4.2e-05, effort_limit=3.3
)
_PCMC = _JointClass(
  armature=0.000136, frictionloss=0.012, damping=4.2e-05, effort_limit=0.5285
)
_MCP = _JointClass(
  armature=0.00129, frictionloss=0.07456, damping=2.38e-05, effort_limit=1.864
)
_PIP = _JointClass(
  armature=0.000415, frictionloss=0.01276, damping=4.06e-06, effort_limit=0.638
)
_DIP = _JointClass(
  armature=2.42e-05,
  frictionloss=0.00378738,
  damping=1.21e-06,
  effort_limit=0.189369,
)


def _actuator(
  joints: str, joint_class: _JointClass, kp: float, kv: float
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=(joints,),
    stiffness=kp,
    damping=kv,
    effort_limit=joint_class.effort_limit,
    armature=joint_class.armature,
    frictionloss=joint_class.frictionloss,
    viscous_damping=joint_class.damping,
  )


_FINGERS = ".*_(index|middle|ring|pinky)"

SHARPA_ACTUATORS = (
  _actuator(".*_thumb_CMC_FE", _CMC, kp=6.95, kv=0.332),
  _actuator(".*_thumb_CMC_AA", _CMC, kp=13.2, kv=0.4672),
  _actuator(".*_thumb_MCP_FE", _MCP, kp=4.76, kv=0.1433),
  _actuator(".*_thumb_MCP_AA", _MCP, kp=6.62, kv=0.169),
  _actuator(".*_thumb_IP", _PIP, kp=0.9, kv=0.03486),
  _actuator(".*_pinky_CMC", _PCMC, kp=1.38, kv=0.02893),
  _actuator(f"{_FINGERS}_MCP_FE", _MCP, kp=4.76, kv=0.1489),
  _actuator(f"{_FINGERS}_MCP_AA", _MCP, kp=6.62, kv=0.1758),
  _actuator(f"{_FINGERS}_PIP", _PIP, kp=0.9, kv=0.03533),
  _actuator(f"{_FINGERS}_DIP", _DIP, kp=0.9, kv=0.008577),
)

##
# Collision config.
##

# Collision geometry follows Menagerie: capsules for the phalanges and thumb,
# cylinders for the joint mounts, and a convex decomposition of the palm. The
# pinky metacarpal uses the convex hull of its visual mesh, since Menagerie's
# collision mesh for it is incomplete. Each fingertip collides through its
# elastomer pad and the rigid shell around it. Mesh hulls are capped at 64
# vertices.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", ".*_fit"),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=0,
  solref={
    ".*_pad_collision": (0.06, 0.9),
    ".*_collision": (0.01, 1.0),
  },
)

##
# Tactile config.
##

# Each fingertip pad carries a taxel grid that runs along the finger (pad x) and
# across it (pad y), with pad z pointing out of the palmar side. The palmar side
# faces -y of the distal phalanx on the right hand and +y on the left.
_PAD_CENTER_X = dict(thumb=0.021, index=0.017, middle=0.017, ring=0.017, pinky=0.017)


def get_fingertip_tactile_cfgs(
  side: Side = "right",
  entity: str = "robot",
  fingers: tuple[str, ...] = tuple(_PAD_CENTER_X),
  resolution: tuple[int, int] = (10, 7),
  debug_vis: bool = False,
) -> tuple[TactileSensorCfg, ...]:
  """One taxel grid per fingertip pad, named "{finger}_tactile"."""
  s = math.sqrt(0.5) if side == "right" else -math.sqrt(0.5)
  return tuple(
    TactileSensorCfg(
      name=f"{finger}_tactile",
      geom=ObjRef("geom", f"{side}_{finger}_pad_collision", entity=entity),
      pos=(_PAD_CENTER_X[finger], 0.0, 0.0),
      quat=(math.sqrt(0.5), s, 0.0, 0.0),
      size=(0.016, 0.010),
      resolution=resolution,
      debug_vis=debug_vis,
    )
    for finger in fingers
  )


##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=SHARPA_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_sharpa_robot_cfg(side: Side = "right") -> EntityCfg:
  return EntityCfg(
    spec_fn=partial(get_spec, side),
    collisions=(FULL_COLLISION,),
    articulation=ARTICULATION,
  )


SHARPA_ACTION_SCALE: dict[str, float] = {}
for a in ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  assert a.effort_limit is not None
  for n in a.target_names_expr:
    SHARPA_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer
  import tyro

  import mjlab
  from mjlab.entity.entity import Entity

  def main(side: Side = "right") -> None:
    viewer.launch(Entity(get_sharpa_robot_cfg(side)).spec.compile())

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
