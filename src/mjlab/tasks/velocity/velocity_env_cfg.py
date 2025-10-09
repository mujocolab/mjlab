"""Velocity tracking task configuration.

This module defines the base configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.manager_term_config import (
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
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig


def create_velocity_env_cfg(
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  foot_friction_geom_names: list[str],
  feet_sensor_names: list[str] | None,
  posture_std: float | dict[str, float] | None,
) -> ManagerBasedRlEnvCfg:
  """Create the base configuration for velocity tracking.

  Args:
    robot_cfg: Robot configuration to add to scene entities.
    action_scale: Scale factor for joint position actions.
    viewer_body_name: Body name for viewer to track.
    foot_friction_geom_names: Geometry names for foot friction randomization.
    feet_sensor_names: Sensor names for feet air time reward.
    posture_std: Standard deviations for posture reward.
  """
  # Scene configuration.
  terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  )
  # Enable curriculum mode for terrain generator.
  if ROUGH_TERRAINS_CFG is not None:
    ROUGH_TERRAINS_CFG.curriculum = True

  # Scene configuration.
  scene_cfg = SceneCfg(
    terrain=terrain_cfg,
    num_envs=1,
    extent=2.0,
    entities={"robot": robot_cfg},
  )

  # Observation configuration.
  policy_obs_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel,
      noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "twist"}
    ),
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
      terms=policy_obs_terms.copy(),
    ),
  }

  # Simulation configuration.
  sim_cfg = SimulationCfg(
    nconmax=140_000,
    njmax=300,
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
      "reset_base": EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
          "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
          "velocity_range": {},
        },
      ),
      "reset_robot_joints": EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
          "position_range": (1.0, 1.0),
          "velocity_range": (0.0, 0.0),
          "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
      ),
      "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
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
    episode_length_s=20.0,
    rewards={
      "track_lin_vel_exp": RewardTermCfg(
        func=mdp.track_lin_vel_exp,
        weight=1.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
      ),
      "track_ang_vel_exp": RewardTermCfg(
        func=mdp.track_ang_vel_exp,
        weight=1.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
      ),
      "pose": RewardTermCfg(
        func=mdp.posture,
        weight=1.0,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
          "std": posture_std or {},
        },
      ),
      "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
      # Unused, only here as an example.
      "air_time": RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
          "asset_name": "robot",
          "threshold_min": 0.05,
          "threshold_max": 0.15,
          "command_name": "twist",
          "command_threshold": 0.05,
          "sensor_names": feet_sensor_names or [],
          "reward_mode": "on_landing",
        },
      ),
    },
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "fell_over": TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
      ),
    },
    commands={
      "twist": mdp.UniformVelocityCommandCfg(
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
      ),
    },
    curriculum={
      "terrain_levels": CurriculumTermCfg(
        func=mdp.terrain_levels_vel, params={"command_name": "twist"}
      ),
      "command_vel": CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
          "command_name": "twist",
          "velocity_stages": [
            {"step": 500 * 24, "range": (-3.0, 3.0)},
          ],
        },
      ),
    },
  )
