"""Unitree G1 rough terrain velocity tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the G1 robot on rough terrain.
"""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.envs import ManagerBasedRlEnvCfg
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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import (
  SCENE_CFG,
  SIM_CFG,
  VIEWER_CONFIG,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def create_unitree_g1_rough_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity tracking configuration."""
  # G1-specific metadata.
  site_names = ["left_foot", "right_foot"]
  geom_names = []
  for i in range(1, 8):
    geom_names.append(f"left_foot{i}_collision")
  for i in range(1, 8):
    geom_names.append(f"right_foot{i}_collision")
  target_foot_height = 0.15

  # Create scene with G1 robot and sensors.
  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": replace(G1_ROBOT_CFG)}

  # Create sensors (G1 uses subtree mode for contact detection)
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  scene.sensors = (feet_ground_cfg, self_collision_cfg)

  # Enable curriculum mode for terrain generator.
  if scene.terrain is not None and scene.terrain.terrain_generator is not None:
    scene.terrain.terrain_generator.curriculum = True

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
    "twist": UniformVelocityCommandCfg(
      asset_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  # Observations
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

  critic_terms = {
    **policy_terms,
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=site_names)},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
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
        "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
      },
    ),
  }

  # Rewards
  rewards = {
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
        "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "command_name": "twist",
        "std_standing": {".*": 0.05},
        "std_walking": {
          r".*hip_pitch.*": 0.3,
          r".*hip_roll.*": 0.15,
          r".*hip_yaw.*": 0.15,
          r".*knee.*": 0.35,
          r".*ankle_pitch.*": 0.25,
          r".*ankle_roll.*": 0.1,
          r".*waist_yaw.*": 0.2,
          r".*waist_roll.*": 0.08,
          r".*waist_pitch.*": 0.1,
          r".*shoulder_pitch.*": 0.15,
          r".*shoulder_roll.*": 0.15,
          r".*shoulder_yaw.*": 0.1,
          r".*elbow.*": 0.15,
          r".*wrist.*": 0.3,
        },
        "std_running": {
          r".*hip_pitch.*": 0.5,
          r".*hip_roll.*": 0.2,
          r".*hip_yaw.*": 0.2,
          r".*knee.*": 0.6,
          r".*ankle_pitch.*": 0.35,
          r".*ankle_roll.*": 0.15,
          r".*waist_yaw.*": 0.3,
          r".*waist_roll.*": 0.08,
          r".*waist_pitch.*": 0.2,
          r".*shoulder_pitch.*": 0.5,
          r".*shoulder_roll.*": 0.2,
          r".*shoulder_yaw.*": 0.15,
          r".*elbow.*": 0.35,
          r".*wrist.*": 0.3,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.02,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision"},
    ),
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
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": target_foot_height,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "target_height": target_foot_height,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
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

  # Terminations
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  # Curriculum
  curriculum = {
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

  # Customize commands.
  twist_cmd = commands["twist"]
  if isinstance(twist_cmd, UniformVelocityCommandCfg):
    twist_cmd.viz.z_offset = 1.15

  # Customize viewer.
  viewer = deepcopy(VIEWER_CONFIG)
  viewer.body_name = "torso_link"

  # Create and return complete config.
  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
    commands=commands,
    curriculum=curriculum,
    sim=SIM_CFG,
    viewer=viewer,
    decimation=4,
    episode_length_s=20.0,
  )


def create_unitree_g1_rough_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain PLAY configuration."""
  cfg = create_unitree_g1_rough_env_cfg()

  # PLAY mode customizations.
  cfg.episode_length_s = int(1e9)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


# Module-level constants for gymnasium registration.
UNITREE_G1_ROUGH_ENV_CFG = create_unitree_g1_rough_env_cfg()
UNITREE_G1_ROUGH_ENV_CFG_PLAY = create_unitree_g1_rough_env_cfg_play()
