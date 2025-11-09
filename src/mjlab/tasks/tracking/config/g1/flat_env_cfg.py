"""Unitree G1 flat terrain tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the G1 robot tracking task on flat terrain.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  CommandTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import (
  SCENE_CFG,
  SIM_CFG,
  VELOCITY_RANGE,
  VIEWER_CONFIG,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise


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

  # Actions
  actions = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=[".*"],
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  # Commands
  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      asset_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      motion_file="",
      anchor_body_name="torso_link",
      body_names=[
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
      ],
    )
  }

  # Observations
  policy_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # Events
  events = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "add_joint_default_pos": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "add",
        "field": "qpos0",
        "ranges": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=[r"^(left|right)_foot[1-7]_collision$"]
        ),
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
      },
    ),
  }

  # Rewards
  rewards = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision"},
    ),
  }

  # Terminations
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": [
          "left_ankle_roll_link",
          "right_ankle_roll_link",
          "left_wrist_yaw_link",
          "right_wrist_yaw_link",
        ],
      },
    ),
  }

  # Viewer
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

  # Disable state estimation observations
  if "policy" in cfg.observations:
    policy_terms = cfg.observations["policy"].terms
    policy_terms.pop("motion_anchor_pos_b", None)
    policy_terms.pop("base_lin_vel", None)

  return cfg


def create_g1_flat_tracking_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking PLAY configuration."""
  cfg = create_g1_flat_tracking_env_cfg()

  # PLAY mode customizations
  if "policy" in cfg.observations:
    cfg.observations["policy"].enable_corruption = False

  if cfg.events is not None:
    cfg.events.pop("push_robot", None)

  # Disable RSI randomization
  if cfg.commands is not None and "motion" in cfg.commands:
    motion_cmd = cfg.commands["motion"]
    if isinstance(motion_cmd, MotionCommandCfg):
      motion_cmd.pose_range = {}
      motion_cmd.velocity_range = {}
      motion_cmd.sampling_mode = "start"

  # Effectively infinite episode length
  cfg.episode_length_s = int(1e9)

  return cfg


def create_g1_flat_tracking_env_cfg_demo() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking DEMO configuration.

  The demo uses a long motion, so we use uniform sampling to see more diversity
  with num_envs > 1.
  """
  cfg = create_g1_flat_tracking_env_cfg_play()

  # Use uniform sampling for demos with long motions
  if cfg.commands is not None and "motion" in cfg.commands:
    motion_cmd = cfg.commands["motion"]
    if isinstance(motion_cmd, MotionCommandCfg):
      motion_cmd.sampling_mode = "uniform"

  return cfg


def create_g1_flat_tracking_no_state_estimation_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat tracking PLAY config without state estimation."""
  cfg = create_g1_flat_tracking_no_state_estimation_env_cfg()

  # PLAY mode customizations
  if "policy" in cfg.observations:
    cfg.observations["policy"].enable_corruption = False

  if cfg.events is not None:
    cfg.events.pop("push_robot", None)

  # Disable RSI randomization
  if cfg.commands is not None and "motion" in cfg.commands:
    motion_cmd = cfg.commands["motion"]
    if isinstance(motion_cmd, MotionCommandCfg):
      motion_cmd.pose_range = {}
      motion_cmd.velocity_range = {}
      motion_cmd.sampling_mode = "start"

  # Effectively infinite episode length
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
