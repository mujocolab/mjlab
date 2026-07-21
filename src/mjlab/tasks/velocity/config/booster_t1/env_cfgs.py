""" Booster T1 task config for velocity tracking.

    uv run list-envs
    uv run play Mjlab-Standing-Flat-Booster-T1 --agent zero
    uv run play Mjlab-Standing-Flat-Booster-T1 --agent random

all work, so you can visually sanity-check the robot loads, stands in a
sane pose, and the leg joints respond to actions -- before we add
velocity commands, a real reward function, and RSL-RL training on top.
"""
import math

from mjlab.asset_zoo.robots.booster_t1.t1_constants import (
  FOOT_BODY_NAMES,
  FOOT_GEOM_NAMES,
  ILLEGAL_CONTACT_BODY_NAMES,
  LEG_JOINT_NAMES,
  get_t1_robot_cfg,
)

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab.asset_zoo.robots.booster_t1.t1_constants import get_t1_robot_cfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp

from mjlab.managers import (
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    ActionTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)

from mjlab.managers.action_manager import ActionTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.utils.noise import GaussianNoiseCfg

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.envs.mdp import events as event_fns, dr
from src.mjlab.sensor import TerrainHeightData, TerrainHeightSensorCfg

from mjlab.sensor import (
  GridPatternCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)


def booster_t1_velocity_env_cfg(num_envs: int = 4096) -> ManagerBasedRlEnvCfg:
  """Create base task for Booster T1"""

  # For my sanity override stuff since mid-migration
  robot_cfg = SceneEntityCfg("robot")
  leg_joints_cfg = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
  trunk_cfg = SceneEntityCfg("robot", body_names=["Trunk"])

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="Trunk", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
  )

  ##
  # Observations
  ##


  # Observation Order
  # 1. projected_gravity (3)
  # 2. base angular velocity (3)
  # 3. velocity commands (3)
  # 4. sin and cos gait phase (2)
  # 6. rel_joint_pos = dof_pose - dof_default_pose (12)
  # 7. dof_vel (12)
  # 8. previous actions (12)

  policy_terms = {
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=GaussianNoiseCfg(mean=0.0, std=0.01)
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      params={"asset_cfg": robot_cfg}

    ),
    "commands": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"}
    ),
    "gait_clock": ObservationTermCfg(
      func=mdp.gait_clock,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "gait_frequency": 1.25
      }
    ),
    "rel_joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": leg_joints_cfg},
      noise=GaussianNoiseCfg(mean=0.0, std=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": leg_joints_cfg},
      noise=GaussianNoiseCfg(mean=0.0, std=0.1)
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action)
  }

  # Critic Order
  # 1. **policy_terms (47)
  # 2. Base mass + CoM offsets (4)
  # 3. Base linear velocity (3)
  # 4. Base height above terrain (1)
  # 5. Applied Force on trunk x 0.1 (3)
  # 6. Applied push torque on trunk x 0.5 (3)

  critic_terms = {
    **policy_terms,
    "base_mass_scaled": ObservationTermCfg(
      func=mdp.com_body_mass,
    ),
    "base_com_offset": ObservationTermCfg(
      func=mdp.com_body_offset,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel,
    ),
    "base_height": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "applied_push_force": ObservationTermCfg(
      func=mdp.base_push_force_priv,
      params={"asset_cfg": trunk_cfg},
      scale=0.1,
    ),
    "applied_push_torque": ObservationTermCfg(
      func=mdp.base_push_torque_priv,
      params={"asset_cfg": trunk_cfg},
      scale=0.5,
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
    )
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.JointPositionActionCfg(
      entity_name="robot",
      actuator_names=LEG_JOINT_NAMES,
      scale=0.25,
      use_default_offset=True,
    )
  }


  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
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

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-1.0, -1.0), "y": (-1.0, 1.0), "yaw": (0, 2 * math.pi)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "pd_gains_dr": EventTermCfg(
        mode="reset",
        func=dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            # VERIFY exact kwarg names -- docs confirm pd_gains randomizes
            # stiffness (kp) and damping (kd) together but don't give the
            # literal param names; check the function's signature/docstring.
            "kp_range": (0.95, 1.05),
            "kd_range": (0.95, 1.05),
            "operation": "scale",
        },
    ),
    "joint_friction_dr": EventTermCfg(
        mode="reset",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "ranges": (0.0, 2.0),
            "operation": "add",
        },
    ),
    "foot_friction_dr": EventTermCfg(
        mode="reset",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
            "ranges": (0.1, 2.0),
            "operation": "add",
        },
    ),
    "com_dr": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="reset",
      params={
        "asset_cfg": trunk_cfg,
        "alpha_range": (-0.2, 0.2),   # ±10% mass/inertia scaling
        "t_range": (-0.1, 0.1),     # ±2 cm COM shift
      },
    ),
    "other_link_mass_dr": EventTermCfg(
        mode="reset",
        func=mdp.dr.pseudo_inertia,
        params={
            "asset_cfg": robot_cfg,
            "alpha_range": (-0.02, 0.02),  # ±2% mass/inertia
        },
    ),
    "push_robot": EventTermCfg(
      func=event_fns.apply_body_impulse,
      mode="step",
      params={
        "force_range": (-0.1, 0.1),
        "torque_range": (-0.5, 0.5),
        "duration_s": (0.05, 0.1),
        "cooldown_s": (2.0, 5.5),
        "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
      },
    ),
  }

  ##
  # Rewards
  ##

  # Shared SceneEntityCfg handles for reward terms
  _robot_cfg      = SceneEntityCfg("robot")
  _leg_joints_cfg = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
  _trunk_cfg      = SceneEntityCfg("robot", body_names=["Trunk"])
  _feet_cfg       = SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES)
  _illegal_cfg    = SceneEntityCfg("robot", body_names=ILLEGAL_CONTACT_BODY_NAMES)

  rewards = {
    # ------------------------------------------------------------------
    # Survival
    # ------------------------------------------------------------------
    "survival": RewardTermCfg(
      func=mdp.is_alive,
      weight=0.25,
    ),

    # ------------------------------------------------------------------
    # Velocity tracking  (built-in mjlab terms from velocity_mdp)
    # ------------------------------------------------------------------
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.5,
      params={
        "command_name": "twist",
        "asset_cfg": _robot_cfg,
        "std": 0.25,
      },
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.5,
      params={
        "command_name": "twist",
        "asset_cfg": _robot_cfg,
        "std": 0.25,
      },
    ),

    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "joint_torque_penalty": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-0.0004,
      params={"asset_cfg": _leg_joints_cfg},
    ),
    "joint_vel_penalty": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-0.0001,
      params={"asset_cfg": _leg_joints_cfg},
    ),
    "joint_acc_penalty": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-0.0000001,
      params={"asset_cfg": _leg_joints_cfg},
    ),
    "action_rate_penalty": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.02,
    ),
    "power": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=-0.002,
      params={"asset_cfg": _leg_joints_cfg},
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  return ManagerBasedRlEnvCfg(
      scene=SceneCfg(
        num_envs=num_envs,
        entities={"robot": get_t1_robot_cfg()},
        terrain=TerrainEntityCfg(terrain_type="plane"),
        sensors=(terrain_scan,)
      ),
      observations=observations,
      actions=actions,
      commands=commands,
      events=events,
      rewards=rewards,
      terminations=terminations,
      episode_length_s=20.0,
      sim=SimulationCfg(mujoco=MujocoCfg()),
      decimation=4,
  )


def booster_t1_velocity_play_env_cfg() -> ManagerBasedRlEnvCfg:
  """Play/eval variant: fewer envs, longer episode, no randomization."""
  cfg = booster_t1_velocity_env_cfg(num_envs=1)
  cfg.episode_length_s = 1_000_000.0
  return cfg
