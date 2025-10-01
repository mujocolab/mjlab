"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.
"""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp  # For general mdp functions.
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


def create_tracking_env_cfg(
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  motion_file: str,
  anchor_body_name: str,
  body_names: list[str],
  base_com_body_name: str,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  joint_position_range: tuple[float, float],
  foot_friction_geom_names: list[str],
  ee_body_names: list[str],
) -> ManagerBasedRlEnvCfg:
  """Create the base configuration for motion tracking.

  Args:
    robot_cfg: Robot configuration to add to scene entities.
    action_scale: Scale factor for joint position actions.
    viewer_body_name: Body name for viewer to track.
    motion_file: Path to motion NPZ file, empty string if no data yet.
    anchor_body_name: Anchor body for motion tracking.
    body_names: Body names to track in motion.
    base_com_body_name: Body name for base center of mass randomization.
    pose_range: Pose randomization ranges.
    velocity_range: Velocity randomization ranges.
    joint_position_range: Joint position randomization range.
    foot_friction_geom_names: Geometry names for foot friction event.
    ee_body_names: End-effector body names for termination.
  """

  # Scene configuration.
  scene_cfg = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    num_envs=1,
    entities={"robot": robot_cfg},
  )

  # Observation configuration.
  policy_obs_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel, noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5)
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2)
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_obs_terms = {
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
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "policy": ObservationGroupCfg(
      concatenate_terms=True,
      enable_corruption=True,
      terms=policy_obs_terms,
    ),
    "critic": ObservationGroupCfg(
      concatenate_terms=True,
      enable_corruption=False,
      terms=critic_obs_terms,
    ),
  }

  # Simulation configuration.
  sim_cfg = SimulationCfg(
    nconmax=150_000,
    njmax=250,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
    ),
  )

  # Viewer configuration.
  viewer_cfg = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name=viewer_body_name,
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )

  return ManagerBasedRlEnvCfg(
    decimation=4,  # 50 Hz control frequency.
    scene=scene_cfg,
    observations=observations,
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        asset_name="robot",
        actuator_names=[".*"],
        scale=action_scale,
        use_default_offset=True,
      ),
    },
    events={
      "reset_scene_to_default": EventTermCfg(
        func=envs_mdp.reset_scene_to_default,
        mode="reset",
      ),
      "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": velocity_range},
      ),
      "base_com": EventTermCfg(
        mode="startup",
        func=mdp.randomize_field,
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names=[base_com_body_name]),
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
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=foot_friction_geom_names),
          "operation": "abs",
          "field": "geom_friction",
          "ranges": (0.3, 1.2),
        },
      ),
    },
    sim=sim_cfg,
    viewer=viewer_cfg,
    episode_length_s=10.0,
    rewards={
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
    },
    terminations={
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
          "body_names": ee_body_names,
        },
      ),
    },
    commands={
      "motion": mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range=pose_range,
        velocity_range=velocity_range,
        joint_position_range=joint_position_range,
        motion_file=motion_file,
        anchor_body_name=anchor_body_name,
        body_names=body_names,
      ),
    },
  )
