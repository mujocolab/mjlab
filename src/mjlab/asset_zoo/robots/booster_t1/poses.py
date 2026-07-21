"""Loads named robot poses (default/stand/crouch/t_pose/...) from poses.yaml.

Pose entries use substring matching against joint names (e.g. "Hip_Pitch"
matches both "Left_Hip_Pitch" and "Right_Hip_Pitch"), with a "default" key
as a catch-all for any joint that doesn't match another substring in that
pose. The shipped poses.yaml only defines leg-related substrings -- see
the ARM_HEAD_WAIST_DEFAULTS note in constants.py for why arm/head/waist
joints are handled separately rather than via this file's catch-all.
"""

from pathlib import Path

import yaml

_POSES_PATH = Path(__file__).parent / "poses.yaml"


def _load_poses_yaml() -> dict[str, dict[str, float]]:
  with open(_POSES_PATH) as f:
    return yaml.safe_load(f)


def resolve_pose(pose_name: str, joint_names: list[str]) -> dict[str, float]:
  """Resolve a named pose to {joint_name: angle} for the given joints.

  For each joint, the longest matching substring key in the pose entry
  wins (so a more specific key would beat a shorter one if both matched
  -- not currently an issue with this file's keys, but future-proofs it).
  Falls back to the pose's "default" value if nothing matches.
  """
  poses = _load_poses_yaml()
  if pose_name not in poses:
    raise KeyError(f"Unknown pose '{pose_name}'. Available: {list(poses.keys())}")

  pose_spec = poses[pose_name]
  fallback = pose_spec.get("default", 0.0)
  substr_keys = [k for k in pose_spec if k != "default"]

  resolved: dict[str, float] = {}
  for joint_name in joint_names:
    matches = [k for k in substr_keys if k in joint_name]
    resolved[joint_name] = pose_spec[max(matches, key=len)] if matches else fallback
  return resolved
