"""Unitree G1 flat terrain tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the G1 robot tracking task on flat terrain.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import (
  SCENE_CFG,
  SIM_CFG,
  VIEWER_CONFIG,
  create_tracking_actions,
  create_tracking_commands,
  create_tracking_events,
  create_tracking_observations,
  create_tracking_rewards,
  create_tracking_terminations,
)


def create_g1_flat_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  # Create scene with G1 robot and sensors.
  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": replace(G1_ROBOT_CFG)}

  # Create self-collision sensor.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  scene.sensors = (self_collision_cfg,)

  # Get base configs.
  actions = create_tracking_actions()
  commands = create_tracking_commands()
  observations = create_tracking_observations()
  events = create_tracking_events()
  rewards = create_tracking_rewards()
  terminations = create_tracking_terminations()

  # Customize actions.
  actions["joint_pos"].scale = G1_ACTION_SCALE

  # Customize commands - set G1-specific body names.
  from mjlab.tasks.tracking.mdp import MotionCommandCfg

  motion_cmd = commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = [
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
  ]

  # Customize events.
  events["foot_friction"].params["asset_cfg"].geom_names = [
    r"^(left|right)_foot[1-7]_collision$"
  ]
  events["base_com"].params["asset_cfg"].body_names = "torso_link"

  # Customize terminations.
  terminations["ee_body_pos"].params["body_names"] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  ]

  # Customize viewer.
  viewer = deepcopy(VIEWER_CONFIG)
  viewer.body_name = "torso_link"

  # Create and return complete config.
  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    events=events,
    sim=SIM_CFG,
    viewer=viewer,
    decimation=4,
    episode_length_s=10.0,
  )


def create_g1_flat_tracking_no_state_estimation_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking config without state estimation.

  This variant disables motion_anchor_pos_b and base_lin_vel observations,
  simulating the lack of state estimation.
  """
  cfg = create_g1_flat_tracking_env_cfg()

  # Disable state estimation observations.
  assert "policy" in cfg.observations
  policy_terms = cfg.observations["policy"].terms
  assert "motion_anchor_pos_b" in policy_terms
  assert "base_lin_vel" in policy_terms
  del policy_terms["motion_anchor_pos_b"]
  del policy_terms["base_lin_vel"]

  return cfg


def create_g1_flat_tracking_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking PLAY configuration."""
  cfg = create_g1_flat_tracking_env_cfg()

  # PLAY mode customizations.
  assert "policy" in cfg.observations
  cfg.observations["policy"].enable_corruption = False

  assert cfg.events is not None
  assert "push_robot" in cfg.events
  del cfg.events["push_robot"]

  # Disable RSI randomization.
  from mjlab.tasks.tracking.mdp import MotionCommandCfg

  assert cfg.commands is not None
  assert "motion" in cfg.commands
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.sampling_mode = "start"

  # Effectively infinite episode length.
  cfg.episode_length_s = int(1e9)

  return cfg


def create_g1_flat_tracking_env_cfg_demo() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking DEMO configuration.

  The demo uses a long motion, so we use uniform sampling to see more diversity
  with num_envs > 1.
  """
  cfg = create_g1_flat_tracking_env_cfg_play()

  # Use uniform sampling for demos with long motions.
  from mjlab.tasks.tracking.mdp import MotionCommandCfg

  assert cfg.commands is not None
  assert "motion" in cfg.commands
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.sampling_mode = "uniform"

  return cfg


def create_g1_flat_tracking_no_state_estimation_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat tracking PLAY config without state estimation."""
  cfg = create_g1_flat_tracking_no_state_estimation_env_cfg()

  # PLAY mode customizations.
  assert "policy" in cfg.observations
  cfg.observations["policy"].enable_corruption = False

  assert cfg.events is not None
  assert "push_robot" in cfg.events
  del cfg.events["push_robot"]

  # Disable RSI randomization.
  from mjlab.tasks.tracking.mdp import MotionCommandCfg

  assert cfg.commands is not None
  assert "motion" in cfg.commands
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.sampling_mode = "start"

  # Effectively infinite episode length.
  cfg.episode_length_s = int(1e9)

  return cfg


# Module-level constants for gymnasium registration.
G1_FLAT_TRACKING_ENV_CFG = create_g1_flat_tracking_env_cfg()
G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG = (
  create_g1_flat_tracking_no_state_estimation_env_cfg()
)
G1_FLAT_TRACKING_ENV_CFG_PLAY = create_g1_flat_tracking_env_cfg_play()
G1_FLAT_TRACKING_ENV_CFG_DEMO = create_g1_flat_tracking_env_cfg_demo()
G1_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG_PLAY = (
  create_g1_flat_tracking_no_state_estimation_env_cfg_play()
)
