"""Booster T1 robot definition for mjlab.

Single source of truth for T1 hardware parameters used in simulation.
Both the task env_cfg and any sanity-check script should import
`get_t1_robot_cfg()` from here rather than hardcoding robot constants.

IMPORTANT -- please verify against your installed mjlab package before
relying on this:

    python -c "from mjlab.entity import EntityCfg, EntityArticulationInfoCfg; help(EntityCfg)"
    python -c "from mjlab.actuator import XmlActuatorCfg; help(XmlActuatorCfg)"

This file was written against mjlab's published docs and real source
snippets (the cartpole tutorial's cartpole_constants.py, the Entity/
Actuator API reference pages) rather than by importing the package
directly, since I don't have network/package access in this session.
There's some chance a field name is slightly off for your exact
mjlab==1.5.0 install. If `uv run list-envs` or `uv run play` throws an
AttributeError/TypeError pointing at this file, send me the traceback
and we'll fix the exact name -- don't assume it's unfixable.

Confirmed API shape (from mjlab docs + a real asset_zoo example):

    EntityCfg(
        spec_fn: Callable[[], mujoco.MjSpec],
        init_state: EntityCfg.InitialStateCfg = ...,
        articulation: EntityArticulationInfoCfg | None = None,
        ...
    )

    EntityArticulationInfoCfg(actuators: tuple[ActuatorCfg, ...])

There is NO `illegal_contact_bodies` / `foot_bodies` / `base_body` field
on EntityCfg -- those concepts belong in the *task* config, as
`SceneEntityCfg("robot", body_names=[...])` passed into reward/
termination terms (contact sensing is wired up per-task, not per-robot).
See booster_t1_velocity_env_cfg.py (sent separately) for where those
actually go.

Joint ordering (12 policy joints, legs only):
    0  Left_Hip_Pitch
    1  Left_Hip_Roll
    2  Left_Hip_Yaw
    3  Left_Knee_Pitch
    4  Left_Ankle_Pitch
    5  Left_Ankle_Roll
    6  Right_Hip_Pitch
    7  Right_Hip_Roll
    8  Right_Hip_Yaw
    9  Right_Knee_Pitch
    10 Right_Ankle_Pitch
    11 Right_Ankle_Roll

The remaining DOFs (head, arms, waist) exist in the MJCF and are wrapped
by the same actuator group below, but the *action manager* in the task
config only drives the 12 leg joints. Un-driven joints keep whatever
ctrl target the entity is initialized with (see init_state.joint_pos
below) -- worth double-checking visually in the zero-agent play run that
the upper body holds its pose rather than sagging to zero on reset.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab.asset_zoo.robots.booster_t1.poses import resolve_pose

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# NOTE: filename is "t1.xml" to match your uploaded file, not
# "T1_locomotion.xml" as the original draft assumed. This should be the
# EDITED copy with hip/knee/ankle kp/kv overrides (see the t1.xml sent
# alongside this file) -- not your original unedited XML.
_ROBOT_DIR = Path(__file__).parent / "xmls"
T1_XML: Path = _ROBOT_DIR / "t1.xml"

assert T1_XML.exists(), f"T1 XML not found at {T1_XML}"


def get_spec() -> mujoco.MjSpec:
  """Load the T1 MJCF and resolve mesh assets."""
  return mujoco.MjSpec.from_file(str(T1_XML))

# Joints not covered by any substring key in poses.yaml (only Hip/Knee/
# Ankle patterns exist there). Kept as an explicit override on top of the
# pose-file leg angles rather than falling through to poses.yaml's
# "default: 0.0", which would flatten the intentional arms-down rest
# pose to a T-pose.
_ARM_HEAD_WAIST_DEFAULTS = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.4,
    "Waist": 0.0,
}

# Root pose from the same keyframe: qpos = "0 0 0.665 1 0 0 0 ...".
_INIT_ROOT_POS = (0.0, 0.0, 0.665)
_INIT_ROOT_ROT = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z), identity


# ---------------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------------
# t1.xml already declares a <position> actuator for every joint, with
# per-joint torque limiting via each joint's `actuatorfrcrange` (e.g.
# Left_Hip_Pitch actuatorfrcrange="-45 45"). The edited copy of the XML
# gives the hip/knee/ankle actuators their own kp/kv (200/5, 200/5, 50/1)
# matching your Isaac Gym config, instead of the uniform kp=75 kv=5 the
# "t1" default class applies to everything. Head/arms/waist stay at the
# XML's 75/5 default -- tune later if needed.
#
# Because the actuators already exist in the XML, we wrap them with
# XmlActuatorCfg rather than creating new BuiltinPositionActuatorCfg
# actuators (which programmatically ADD <position> elements and would
# duplicate the ones already in the file).
_ALL_ACTUATORS = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
)

def _build_default_joint_angles(pose_name: str = "default") -> dict[str, float]:
  leg_angles = resolve_pose(pose_name, LEG_JOINT_NAMES)
  return {**_ARM_HEAD_WAIST_DEFAULTS, **leg_angles}


def get_t1_robot_cfg(pose_name: str = "default") -> EntityCfg:
  """Return a fresh EntityCfg for the Booster T1.

  pose_name selects a leg pose from poses.yaml ("default", "stand",
  "crouch", "t_pose", ...). Arm/head/waist joints always use the fixed
  rest pose in _ARM_HEAD_WAIST_DEFAULTS regardless of pose_name.
  """
  return EntityCfg(
      spec_fn=get_spec,
      init_state=EntityCfg.InitialStateCfg(
          pos=_INIT_ROOT_POS,
          rot=_INIT_ROOT_ROT,
          joint_pos=_build_default_joint_angles(pose_name),
      ),
      articulation=_ALL_ACTUATORS,
  )

# Convenience name lists for use in the task config (observations, the
# action manager's joint scope, and SceneEntityCfg body_names for
# contact termination/reward terms). These are plain Python data, not
# EntityCfg fields -- pass them into the task config directly.

LEG_JOINT_NAMES = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

# Body names verified against the t1.xml you attached (these DO match
# your Isaac Gym names like "Trunk"/"H1"/"AL1" -- your XML happens to
# use the same names for those -- but the ankle/foot bodies are named
# differently than a naive guess: the ankle-pitch link is
# "Ankle_Cross_Left/Right", and the terminal foot link, which also
# carries the ankle-roll joint, is "left_foot_link"/"right_foot_link").
BASE_BODY_NAME = "Trunk"

FOOT_BODY_NAMES = ["left_foot_link", "right_foot_link"]
FOOT_GEOM_NAMES = ["left_foot_collision", "right_foot_collision"]

# Bodies that should trigger an illegal-contact termination/penalty if
# they touch the ground: everything between the trunk and the ankle,
# i.e. everything except the feet.
ILLEGAL_CONTACT_BODY_NAMES = [
    "Trunk", "H1", "H2",
    "AL1", "AL2", "AL3", "left_hand_link",
    "AR1", "AR2", "AR3", "right_hand_link",
    "Waist",
    "Hip_Pitch_Left", "Hip_Roll_Left", "Hip_Yaw_Left", "Shank_Left",
    "Ankle_Cross_Left",
    "Hip_Pitch_Right", "Hip_Roll_Right", "Hip_Yaw_Right", "Shank_Right",
    "Ankle_Cross_Right",
]
