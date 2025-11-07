"""Velocity tracking task configuration.

This module defines the base configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.

After refactoring: This module provides factory functions that return dictionaries
for each config section. Robot-specific configs compose these to create complete
ManagerBasedRlEnvCfg instances.
"""

import math

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  CommandTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Scene.
##

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  ),
  num_envs=1,
  extent=2.0,
)

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
  njmax=300,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)

##
# MDP - Factory functions that return dictionaries.
##


def create_velocity_actions() -> dict[str, JointPositionActionCfg]:
  """Create base velocity task actions.

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


def create_velocity_commands() -> dict[str, CommandTermCfg]:
  """Create base velocity task commands."""
  return {
    "twist": mdp.UniformVelocityCommandCfg(
      asset_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }


def create_velocity_observations() -> dict[str, ObservationGroupCfg]:
  """Create base velocity task observations.

  Robot-specific configs can modify terms (e.g., set site_names for foot_height).
  """
  # Policy observation terms.
  policy_terms = {
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
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }

  # Critic observation terms (policy + privileged info)
  critic_terms = {
    **policy_terms,  # Include all policy observations
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=[])  # Override in robot cfg.
      },
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={
        "sensor_name": "feet_ground_contact",
      },
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
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


def create_velocity_events() -> dict[str, EventTermCfg]:
  """Create base velocity task events.

  Robot-specific configs can modify params (e.g., set geom_names for foot_friction).
  """
  return {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
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


def create_velocity_rewards() -> dict[str, RewardTermCfg]:
  """Create base velocity task rewards.

  Robot-specific configs can modify weights and params (e.g., pose std values).
  """
  return {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=[]),  # Override in robot cfg.
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "command_name": "twist",
        "std_standing": {},  # Override in robot cfg.
        "std_walking": {},  # Override in robot cfg.
        "std_running": {},  # Override in robot cfg.
        "walking_threshold": 0.05,  # m/s
        "running_threshold": 1.5,  # m/s
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=[]),  # Override in robot cfg.
      },
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.02,
      params={
        "sensor_name": "robot/root_angmom",
      },
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision"},
    ),
    # Rewards feet being airborne for 0.05-0.5 seconds.
    # Lift your feet off the ground and keep them up for a reasonable amount of time.
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    # Guide the foot height during the swing phase.
    # Large penalty when foot is moving fast and far from target height.
    # This is a dense reward.
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=[]),
      },
    ),
    # Tracks peak height during swing. Did you actually reach 0.1m at some point?
    # This is a sparse reward, only evaluated at landing.
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=[]),  # Override in robot cfg.
      },
    ),
    # Don't slide when foot is on ground.
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=[]),  # Override in robot cfg.
      },
    ),
    # Encourage soft landings.
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
  }


def create_velocity_terminations() -> dict[str, TerminationTermCfg]:
  """Create base velocity task terminations."""
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "nonfoot_ground_touch"},
    ),
  }


def create_velocity_curriculum() -> dict[str, CurriculumTermCfg]:
  """Create base velocity task curriculum."""
  return {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    ),
  }
