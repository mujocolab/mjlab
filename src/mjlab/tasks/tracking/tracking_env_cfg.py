"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448

After refactoring: This module provides factory functions that return dictionaries
for each config section. Robot-specific configs compose these to create complete
ManagerBasedRlEnvCfg instances.
"""

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
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}

##
# Scene.
##

SCENE_CFG = SceneCfg(terrain=TerrainImporterCfg(terrain_type="plane"), num_envs=1)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",  # Override in robot cfg.
  distance=3.0,
  elevation=-5.0,
  azimuth=90.0,
)

##
# Simulation.
##

SIM_CFG = SimulationCfg(
  nconmax=35,
  njmax=250,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)

##
# MDP - Factory functions that return dictionaries.
##


def create_tracking_commands() -> dict[str, CommandTermCfg]:
  """Create base tracking task commands.

  Returns dictionary mapping command names to CommandTermCfg instances.
  Robot-specific configs can modify params (e.g., set motion_file, body_names).
  """
  return {
    "motion": mdp.MotionCommandCfg(
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
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=[],
    )
  }


def create_tracking_actions() -> dict[str, JointPositionActionCfg]:
  """Create base tracking task actions.

  Returns dictionary mapping action names to ActionTermCfg instances.
  Robot-specific configs can modify the returned dict (e.g., change scale).
  """
  return {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=[".*"],
      scale=0.5,
      use_default_offset=True,
    )
  }


def create_tracking_observations() -> dict[str, ObservationGroupCfg]:
  """Create base tracking task observations.

  Returns dictionary with 'policy' and 'critic' observation groups.
  Robot-specific configs can modify terms (e.g., set observation terms to None).
  """
  # Policy observation terms.
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

  # Critic observation terms (privileged info - no noise)
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

  return {
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


def create_tracking_events() -> dict[str, EventTermCfg]:
  """Create base tracking task events.

  Returns dictionary mapping event names to EventTermCfg instances.
  Robot-specific configs can modify params (e.g., set geom_names for foot_friction).
  """
  return {
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
        "asset_cfg": SceneEntityCfg("robot", body_names=[]),  # Override in robot cfg.
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
        "asset_cfg": SceneEntityCfg("robot", geom_names=[]),  # Override in robot cfg.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
      },
    ),
  }


def create_tracking_rewards() -> dict[str, RewardTermCfg]:
  """Create base tracking task rewards.

  Returns dictionary mapping reward names to RewardTermCfg instances.
  Robot-specific configs can modify weights and params.
  """
  return {
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


def create_tracking_terminations() -> dict[str, TerminationTermCfg]:
  """Create base tracking task terminations.

  Returns dictionary mapping termination names to TerminationTermCfg instances.
  Robot-specific configs can modify params (e.g., set body_names for ee_body_pos).
  """
  return {
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
        "body_names": [],  # Override in robot cfg.
      },
    ),
  }
