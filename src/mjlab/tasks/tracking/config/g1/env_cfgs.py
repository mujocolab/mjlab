"""Unitree G1 flat terrain tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the G1 robot tracking task on flat terrain.
"""

from copy import deepcopy

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import create_tracking_env_cfg
from mjlab.utils.retval import retval


@retval
def UNITREE_G1_FLAT_TRACKING_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  return create_tracking_env_cfg(
    robot_cfg=get_g1_robot_cfg(),
    action_scale=G1_ACTION_SCALE,
    viewer_body_name="torso_link",
    motion_file="",
    anchor_body_name="torso_link",
    body_names=(
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "torso_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    ),
    foot_friction_geom_names=(r"^(left|right)_foot[1-7]_collision$",),
    ee_body_names=(
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    ),
    base_com_body_name="torso_link",
    sensors=(self_collision_cfg,),
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range={
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-0.78, 0.78),
    },
    joint_position_range=(-0.1, 0.1),
  )


@retval
def UNITREE_G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking config without state estimation.

  This variant disables motion_anchor_pos_b and base_lin_vel observations,
  simulating the lack of state estimation.
  """
  cfg = deepcopy(UNITREE_G1_FLAT_TRACKING_ENV_CFG)
  assert "policy" in cfg.observations
  cfg.observations["policy"].terms.pop("motion_anchor_pos_b")
  cfg.observations["policy"].terms.pop("base_lin_vel")
  return cfg
