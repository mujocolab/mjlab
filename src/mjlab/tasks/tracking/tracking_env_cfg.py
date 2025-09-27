"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.
"""

from copy import deepcopy

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp  # For general mdp functions
from mjlab.managers.manager_term_config import (
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
from mjlab.utils.noise import UniformNoiseCfg
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


##
# MDP.
##


def create_tracking_env_cfg(
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  motion_file: str,
  reference_body: str,
  body_names: list[str],
  foot_friction_geom_names: list[str],
  ee_body_names: list[str],
  viewer_distance: float = 3.0,
  viewer_elevation: float = -5.0,
  viewer_azimuth: float = 90.0,
  pose_range: dict[str, tuple[float, float]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  joint_position_range: tuple[float, float] = (-0.1, 0.1),
) -> ManagerBasedRlEnvCfg:
  """Create the base configuration for motion tracking.

  Args:
    robot_cfg: Robot configuration to add to scene entities.
    action_scale: Scale factor for joint position actions.
    viewer_body_name: Body name for viewer to track.
    motion_file: Path to motion NPZ file, empty string if no data yet.
    reference_body: Reference body for motion tracking.
    body_names: Body names to track in motion.
    foot_friction_geom_names: Geometry names for foot friction event.
    ee_body_names: End-effector body names for termination.
    viewer_distance: Distance from viewer to tracked body.
    viewer_elevation: Elevation angle for viewer.
    viewer_azimuth: Azimuth angle for viewer.
    pose_range: Pose randomization ranges.
    velocity_range: Velocity randomization ranges.
    joint_position_range: Joint position randomization range.
  """

  # Default pose and velocity ranges.
  if pose_range is None:
    pose_range = {
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    }
  if velocity_range is None:
    velocity_range = VELOCITY_RANGE

  # Build policy observation terms.
  policy_obs_terms = dict(
    command=ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    motion_ref_pos_b=ObservationTermCfg(
      func=mdp.motion_ref_pos_b,
      params={"command_name": "motion"},
      noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
    ),
    motion_ref_ori_b=ObservationTermCfg(
      func=mdp.motion_ref_ori_b,
      params={"command_name": "motion"},
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    ),
    base_lin_vel=ObservationTermCfg(
      func=mdp.base_lin_vel,
      noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
    ),
    base_ang_vel=ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    ),
    joint_pos=ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    ),
    joint_vel=ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
    ),
    actions=ObservationTermCfg(func=mdp.last_action),
  )

  # Build privileged observation terms for critic.
  privileged_obs_terms = policy_obs_terms.copy()
  privileged_obs_terms.update(
    dict(
      # Additional privileged observations for critic.
      body_pos=ObservationTermCfg(
        func=mdp.robot_body_pos_b, params={"command_name": "motion"}
      ),
      body_ori=ObservationTermCfg(
        func=mdp.robot_body_ori_b, params={"command_name": "motion"}
      ),
    )
  )

  # Create scene with robot entity.
  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": robot_cfg}

  cfg = ManagerBasedRlEnvCfg(
    decimation=4,  # 50 Hz control frequency
    episode_length_s=10.0,
    scene=scene,
    sim=SimulationCfg(
      nconmax=150_000,
      njmax=250,
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
      azimuth=viewer_azimuth,
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
      motion=mdp.MotionCommandCfg(
        class_type=mdp.MotionCommand,
        resampling_time_range=(1e9, 1e9),  # No resampling by default
        debug_vis=True,
        motion_file=motion_file,
        reference_body=reference_body,
        body_names=body_names,
        asset_name="robot",
        pose_range=pose_range,
        velocity_range=velocity_range,
        joint_position_range=joint_position_range,
      ),
    ),
    observations=dict(
      policy=ObservationGroupCfg(
        concatenate_terms=True,
        concatenate_dim=-1,
        enable_corruption=True,
        terms=policy_obs_terms,
      ),
      critic=ObservationGroupCfg(
        concatenate_terms=True,
        concatenate_dim=-1,
        enable_corruption=False,
        terms=privileged_obs_terms,
      ),
    ),
    events=dict(
      push_robot=EventTermCfg(
        func=envs_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
      ),
      add_joint_default_pos=EventTermCfg(
        mode="startup",
        func=envs_mdp.randomize_field,
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "operation": "add",
          "field": "qpos0",
          "ranges": (-0.01, 0.01),
        },
      ),
      foot_friction=EventTermCfg(
        mode="startup",
        func=envs_mdp.randomize_field,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=foot_friction_geom_names),
          "operation": "abs",
          "field": "geom_friction",
          "ranges": (0.3, 1.2),
        },
      ),
    ),
    rewards=dict(
      # Motion tracking rewards.
      motion_global_root_pos=RewardTermCfg(
        func=mdp.motion_global_ref_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
      ),
      motion_global_root_ori=RewardTermCfg(
        func=mdp.motion_global_ref_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
      ),
      motion_body_pos=RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
      ),
      motion_body_ori=RewardTermCfg(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
      ),
      motion_body_lin_vel=RewardTermCfg(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
      ),
      motion_body_ang_vel=RewardTermCfg(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
      ),
      # Non-motion rewards.
      action_rate_l2=RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
      joint_limit=RewardTermCfg(
        func=envs_mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
      ),
      self_collisions=RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-10.0,
        params={"sensor_name": "self_collision"},
      ),
    ),
    terminations=dict(
      time_out=TerminationTermCfg(func=mdp.time_out, time_out=True),
      ref_pos=TerminationTermCfg(
        func=mdp.bad_ref_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
      ),
      ref_ori=TerminationTermCfg(
        func=mdp.bad_ref_ori,
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "command_name": "motion",
          "threshold": 0.8,
        },
      ),
      ee_body_pos=TerminationTermCfg(
        func=mdp.bad_motion_body_pos_z_only,
        params={
          "command_name": "motion",
          "threshold": 0.25,
          "body_names": ee_body_names,
        },
      ),
    ),
    curriculum=None,
  )

  return cfg
