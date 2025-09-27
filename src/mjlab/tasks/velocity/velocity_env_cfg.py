"""Velocity tracking task configuration.

This module defines the base configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math
from copy import deepcopy

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import CurriculumTermCfg as CurrTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
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

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  ),
  num_envs=1,
  extent=2.0,
)


##
# MDP.
##


def create_locomotion_velocity_env_cfg(
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  air_time_sensor_names: list[str],
  foot_friction_geom_names: list[str],
  foot_clearance_geom_names: list[str],
  pose_l2_std: dict[str, float],
  viewer_distance: float = 3.0,
  viewer_elevation: float = -5.0,
  command_viz_z_offset: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the base configuration for locomotion velocity tracking.

  Args:
    robot_cfg: Robot configuration to add to scene entities.
    action_scale: Scale factor for joint position actions.
    viewer_body_name: Body name for viewer to track.
    air_time_sensor_names: Sensor names for air time reward.
    foot_friction_geom_names: Geometry names for foot friction event.
    foot_clearance_geom_names: Geometry names for foot clearance reward.
    pose_l2_std: Standard deviation dict for pose L2 reward.
    viewer_distance: Distance from viewer to tracked body (default: 3.0).
    viewer_elevation: Elevation angle for viewer (default: -5.0).
    command_viz_z_offset: Z offset for command visualization (optional).
  """
  # Build policy observation terms.
  policy_obs_terms = dict(
    base_lin_vel=ObsTerm(
      func=mdp.base_lin_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    base_ang_vel=ObsTerm(
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    projected_gravity=ObsTerm(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    joint_pos=ObsTerm(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    joint_vel=ObsTerm(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    actions=ObsTerm(func=mdp.last_action),
    command=ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"}),
  )

  # Create scene with robot entity.
  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": robot_cfg}
  # Enable curriculum mode for terrain generator.
  if scene.terrain is not None:
    if scene.terrain.terrain_generator is not None:
      scene.terrain.terrain_generator.curriculum = True

  cfg = ManagerBasedRlEnvCfg(
    decimation=4,  # 50 Hz control frequency
    episode_length_s=20.0,
    scene=scene,
    sim=SimulationCfg(
      nconmax=140_000,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      asset_name="robot",
      body_name=viewer_body_name,
      distance=viewer_distance,
      elevation=viewer_elevation,
      azimuth=90.0,
    ),
    actions=dict(
      joint_pos=mdp.JointPositionActionCfg(
        asset_name="robot",
        actuator_names=[".*"],
        scale=action_scale,
        use_default_offset=True,
      ),
    ),
    commands=dict(
      twist=mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(-1.0, 1.0),
          lin_vel_y=(-0.5, 0.5),
          ang_vel_z=(-1.0, 1.0),
          heading=(-math.pi, math.pi),
        ),
        viz=mdp.UniformVelocityCommandCfg.VizCfg(
          z_offset=command_viz_z_offset if command_viz_z_offset is not None else 0.2,
          scale=0.75,
        ),
      ),
    ),
    observations=dict(
      policy=ObsGroup(
        concatenate_terms=True,
        concatenate_dim=-1,
        enable_corruption=True,
        terms=policy_obs_terms,
      ),
      critic=ObsGroup(
        concatenate_terms=True,
        concatenate_dim=-1,
        enable_corruption=False,
        terms=policy_obs_terms.copy(),
      ),
    ),
    events=dict(
      reset_base=EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
          "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
          "velocity_range": {},
        },
      ),
      reset_robot_joints=EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
          "position_range": (1.0, 1.0),
          "velocity_range": (0.0, 0.0),
          "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
      ),
      push_robot=EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
      ),
      foot_friction=EventTerm(
        mode="startup",
        func=mdp.randomize_field,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=foot_friction_geom_names),
          "operation": "abs",
          "field": "geom_friction",
          "ranges": (0.3, 1.2),
        },
      ),
    ),
    rewards=dict(
      # Primary task rewards.
      track_lin_vel_xy_exp=RewardTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
      ),
      track_ang_vel_z_exp=RewardTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
      ),
      # Stability penalties.
      ang_vel_xy_l2=RewardTerm(func=mdp.ang_vel_xy_l2, weight=-0.05),
      # Smoothness penalties.
      action_rate_l2=RewardTerm(func=mdp.action_rate_l2, weight=-0.01),
      smoothness=RewardTerm(func=mdp.gait_smoothness, weight=-0.0001),
      # Efficiency penalties.
      cost_of_transport=RewardTerm(
        func=mdp.cost_of_transport,
        weight=-0.1,
        params={
          "asset_name": "robot",
          "min_velocity": 0.1,
          "normalize_by_mass": False,
          "power_scale": 0.001,
        },
      ),
      # Posture and limits.
      pose_l2=RewardTerm(
        func=mdp.posture,
        weight=-0.5,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
          "std": pose_l2_std,
        },
      ),
      dof_pos_limits=RewardTerm(func=mdp.joint_pos_limits, weight=-1.0),
      # Gait shaping.
      air_time=RewardTerm(
        func=mdp.feet_air_time,
        weight=0.3,
        params={
          "asset_name": "robot",
          "threshold_min": 0.05,
          "threshold_max": 0.15,
          "command_name": "twist",
          "command_threshold": 0.05,
          "sensor_names": air_time_sensor_names,
          "reward_mode": "on_landing",
        },
      ),
      foot_clearance=RewardTerm(
        func=mdp.foot_clearance_reward,
        weight=0.5,
        params={
          "std": 0.05,
          "tanh_mult": 2.0,
          "target_height": 0.1,
          "asset_cfg": SceneEntityCfg("robot", geom_names=foot_clearance_geom_names),
        },
      ),
    ),
    terminations=dict(
      time_out=DoneTerm(func=mdp.time_out, time_out=True),
      fell_over=DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
      ),
    ),
    curriculum=dict(
      terrain_levels=CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"command_name": "twist"},
      ),
    ),
  )

  return cfg
