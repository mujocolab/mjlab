"""Default pose shared by the G1 football training stages."""

from mjlab.entity import EntityCfg


def get_isaaclab_default_keyframe() -> EntityCfg.InitialStateCfg:
  """Return a fresh copy of the IsaacLab G1 football default pose."""
  return EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.78),
    joint_pos={
      ".*_hip_pitch_joint": -0.1,
      ".*_knee_joint": 0.3,
      ".*_ankle_pitch_joint": -0.2,
      ".*_shoulder_pitch_joint": 0.35,
      "left_shoulder_roll_joint": 0.18,
      "right_shoulder_roll_joint": -0.18,
      ".*_elbow_joint": 0.6,
    },
    joint_vel={".*": 0.0},
  )
