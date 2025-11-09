"""Unitree G1 flat terrain tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the G1 robot tracking task on flat terrain.
"""

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import create_tracking_env_cfg


def create_g1_flat_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg = create_tracking_env_cfg(
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
  )

  assert cfg.scene is not None
  cfg.scene.sensors = (self_collision_cfg,)

  return cfg


def create_g1_flat_tracking_no_state_estimation_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking config without state estimation.

  This variant disables motion_anchor_pos_b and base_lin_vel observations,
  simulating the lack of state estimation.
  """
  cfg = create_g1_flat_tracking_env_cfg()

  assert "policy" in cfg.observations
  policy_terms = cfg.observations["policy"].terms
  policy_terms.pop("motion_anchor_pos_b", None)
  policy_terms.pop("base_lin_vel", None)

  return cfg


def create_g1_flat_tracking_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking PLAY configuration."""
  cfg = create_g1_flat_tracking_env_cfg()

  assert "policy" in cfg.observations
  cfg.observations["policy"].enable_corruption = False

  assert cfg.events is not None
  cfg.events.pop("push_robot", None)

  # Disable RSI randomization.
  assert cfg.commands is not None and "motion" in cfg.commands
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.sampling_mode = "start"

  cfg.episode_length_s = int(1e9)

  return cfg


def create_g1_flat_tracking_env_cfg_demo() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking DEMO configuration.

  The demo uses a long motion, so we use uniform sampling to see more diversity
  with num_envs > 1.
  """
  cfg = create_g1_flat_tracking_env_cfg_play()

  assert cfg.commands is not None and "motion" in cfg.commands
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.sampling_mode = "uniform"

  return cfg


def create_g1_flat_tracking_no_state_estimation_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat tracking PLAY config without state estimation."""
  cfg = create_g1_flat_tracking_no_state_estimation_env_cfg()

  assert "policy" in cfg.observations
  cfg.observations["policy"].enable_corruption = False

  assert cfg.events is not None
  cfg.events.pop("push_robot", None)

  # Disable RSI randomization.
  assert cfg.commands is not None and "motion" in cfg.commands
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.sampling_mode = "start"

  cfg.episode_length_s = int(1e9)

  return cfg


G1_FLAT_TRACKING_ENV_CFG = create_g1_flat_tracking_env_cfg()
G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG = (
  create_g1_flat_tracking_no_state_estimation_env_cfg()
)
G1_FLAT_TRACKING_ENV_CFG_PLAY = create_g1_flat_tracking_env_cfg_play()
G1_FLAT_TRACKING_ENV_CFG_DEMO = create_g1_flat_tracking_env_cfg_demo()
G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG_PLAY = (
  create_g1_flat_tracking_no_state_estimation_env_cfg_play()
)
