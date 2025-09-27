"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.
"""

from copy import deepcopy

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp  # For general mdp functions
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
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


##
# MDP.
##

SIM_CFG = SimulationCfg(
  nconmax=140_000,
  njmax=300,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)


def create_tracking_env_cfg(
  # Robot-specific parameters (required)
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  # Viewer settings (required)
  viewer_body_name: str,
  # Motion command parameters (required for tracking)
  motion_file: str,  # Empty string if no motion data yet
  reference_body: str,
  body_names: list[str],
  # Event parameters (required)
  foot_friction_geom_names: list[str],
  # Termination parameters (required)
  ee_body_names: list[str],
  # Optional parameters with sensible defaults
  viewer_distance: float = 3.0,
  viewer_elevation: float = -5.0,
  viewer_azimuth: float = 90.0,
  pose_range: dict[str, tuple[float, float]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  joint_position_range: tuple[float, float] = (-0.1, 0.1),
) -> ManagerBasedRlEnvCfg:
  """Create the base configuration for motion tracking.

  Args:
    robot_cfg: Robot configuration to add to scene entities (required).
    action_scale: Scale factor for joint position actions (required).
    viewer_body_name: Body name for viewer to track (required).
    motion_file: Path to motion NPZ file, empty string if no data yet (required).
    reference_body: Reference body for motion tracking (required).
    body_names: Body names to track in motion (required).
    foot_friction_geom_names: Geometry names for foot friction event (required).
    ee_body_names: End-effector body names for termination (required).
    viewer_distance: Distance from viewer to tracked body (default: 3.0).
    viewer_elevation: Elevation angle for viewer (default: -5.0).
    viewer_azimuth: Azimuth angle for viewer (default: 90.0).
    pose_range: Pose randomization ranges (optional, defaults set if None).
    velocity_range: Velocity randomization ranges (optional, defaults set if None).
    joint_position_range: Joint position randomization range (default: (-0.1, 0.1)).
  """

  # Default pose and velocity ranges
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

  # Build rewards dict - always include motion rewards (matching original)
  rewards = dict(
    # Motion tracking rewards (always present, like original)
    motion_global_root_pos=RewTerm(
      func=mdp.motion_global_ref_position_error_exp,
      weight=0.5,
      params=dict(command_name="motion", std=0.3),
    ),
    motion_global_root_ori=RewTerm(
      func=mdp.motion_global_ref_orientation_error_exp,
      weight=0.5,
      params=dict(command_name="motion", std=0.4),
    ),
    motion_body_pos=RewTerm(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params=dict(command_name="motion", std=0.3),
    ),
    motion_body_ori=RewTerm(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params=dict(command_name="motion", std=0.4),
    ),
    motion_body_lin_vel=RewTerm(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params=dict(command_name="motion", std=1.0),
    ),
    motion_body_ang_vel=RewTerm(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params=dict(command_name="motion", std=3.14),
    ),
    # Non-motion rewards
    action_rate_l2=RewTerm(func=mdp.action_rate_l2, weight=-0.1),
    joint_limit=RewTerm(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params=dict(asset_cfg=SceneEntityCfg("robot", joint_names=[".*"])),
    ),
    self_collisions=RewTerm(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params=dict(sensor_name="self_collision"),
    ),
  )

  # Build events dict
  events = dict(
    push_robot=EventTerm(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params=dict(velocity_range=VELOCITY_RANGE),
    ),
    add_joint_default_pos=EventTerm(
      mode="startup",
      func=envs_mdp.randomize_field,
      params=dict(
        asset_cfg=SceneEntityCfg("robot"),
        operation="add",
        field="qpos0",
        ranges=(-0.01, 0.01),
      ),
    ),
  )

  # Add base_com event if we have the function
  # Note: The original used body ipos randomization, but we'll skip this
  # since the exact implementation varies

  # Build terminations dict - always include motion terminations (matching original)
  terminations = dict(
    time_out=DoneTerm(func=mdp.time_out, time_out=True),
    ref_pos=DoneTerm(
      func=mdp.bad_ref_pos_z_only,
      params=dict(command_name="motion", threshold=0.25),
    ),
    ref_ori=DoneTerm(
      func=mdp.bad_ref_ori,
      params=dict(
        asset_cfg=SceneEntityCfg("robot"),
        command_name="motion",
        threshold=0.8,
      ),
    ),
  )

  cfg = ManagerBasedRlEnvCfg(
    decimation=4,  # 50 Hz control frequency
    episode_length_s=5.0,
    scene=deepcopy(SCENE_CFG),
    sim=deepcopy(SIM_CFG),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      asset_name="robot",
      body_name=viewer_body_name,
      distance=viewer_distance,
      elevation=viewer_elevation,
      azimuth=viewer_azimuth,
    ),
    # Actions
    actions=dict(
      joint_pos=mdp.JointPositionActionCfg(
        asset_name="robot",
        actuator_names=[".*"],
        scale=action_scale,
        use_default_offset=True,
      ),
    ),
    # Commands - always configure motion command (matching original)
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
    # Observations
    observations=dict(
      policy=ObsGroup(
        concatenate_terms=True,
        concatenate_dim=-1,
        enable_corruption=True,
        terms=dict(
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
        ),
      ),
    ),
    # Events
    events=events,
    # Rewards
    rewards=rewards,
    # Terminations
    terminations=terminations,
    # Curriculum (none for tracking)
    curriculum=None,
  )

  # Add robot config
  cfg.scene.entities = {"robot": robot_cfg}

  # Configure foot friction event
  cfg.events["foot_friction"] = EventTerm(
    mode="startup",
    func=envs_mdp.randomize_field,
    params=dict(
      asset_cfg=SceneEntityCfg("robot", geom_names=foot_friction_geom_names),
      operation="abs",
      field="geom_friction",
      ranges=(0.3, 1.2),
    ),
  )

  # Configure end-effector termination
  cfg.terminations["ee_body_pos"] = DoneTerm(
    func=mdp.bad_motion_body_pos_z_only,
    params=dict(
      command_name="motion",
      threshold=0.25,
      body_names=ee_body_names,
    ),
  )

  # Add motion-specific observations (always present, matching original)
  cfg.observations["policy"].terms.update(
    dict(
      motion_ref_pos=ObsTerm(
        func=mdp.motion_ref_pos_b, params=dict(command_name="motion")
      ),
      motion_ref_ori=ObsTerm(
        func=mdp.motion_ref_ori_b, params=dict(command_name="motion")
      ),
      body_pos=ObsTerm(func=mdp.robot_body_pos_b, params=dict(command_name="motion")),
      body_ori=ObsTerm(func=mdp.robot_body_ori_b, params=dict(command_name="motion")),
    )
  )

  # Add critic observations (privileged)
  cfg.observations["critic"] = ObsGroup(
    concatenate_terms=True,
    concatenate_dim=-1,
    enable_corruption=False,
    terms=cfg.observations["policy"].terms.copy(),
  )

  return cfg
