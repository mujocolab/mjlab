"""Velocity tracking task configuration.

This module defines the base configuration for velocity tracking tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import CurriculumTermCfg as CurrTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
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
# MDP.
##


@dataclass
class ActionCfg:
  joint_pos: mdp.JointPositionActionCfg = term(
    mdp.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


@dataclass
class CommandsCfg:
  twist: mdp.UniformVelocityCommandCfg = term(
    mdp.UniformVelocityCommandCfg,
    asset_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.1,
    rel_heading_envs=1.0,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=mdp.UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  )


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    base_lin_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    )

    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)
    command: ObsTerm = term(
      ObsTerm, func=mdp.generated_commands, params={"command_name": "twist"}
    )

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class PrivilegedCfg(PolicyCfg):
    foot_height: ObsTerm = term(
      ObsTerm,
      func=mdp.foot_height,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=[])  # Override in robot cfg.
      },
    )
    foot_air_time: ObsTerm = term(
      ObsTerm,
      func=mdp.foot_air_time,
      params={
        "sensor_name": "feet_ground_contact",
      },
    )
    foot_contact: ObsTerm = term(
      ObsTerm,
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    )

    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)


@dataclass
class EventCfg:
  reset_base: EventTerm = term(
    EventTerm,
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
      "velocity_range": {},
    },
  )
  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
      "position_range": (1.0, 1.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    },
  )
  push_robot: EventTerm | None = term(
    EventTerm,
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
  )
  foot_friction: EventTerm = term(
    EventTerm,
    mode="startup",
    func=mdp.randomize_field,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=[]),  # Override in robot cfg.
      "operation": "abs",
      "field": "geom_friction",
      "ranges": (0.3, 1.2),
    },
  )


@dataclass
class RewardCfg:
  track_linear_velocity: RewardTerm = term(
    RewardTerm,
    func=mdp.track_linear_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )
  track_angular_velocity: RewardTerm = term(
    RewardTerm,
    func=mdp.track_angular_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )
  upright: RewardTerm = term(
    RewardTerm,
    func=mdp.flat_orientation,
    weight=1.0,
    params={"std": math.sqrt(0.1)},
  )
  pose: RewardTerm = term(
    RewardTerm,
    func=mdp.variable_posture,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
      "std_standing": {},  # Override in robot cfg.
      "std_moving": {},  # Override in robot cfg.
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  dof_pos_limits: RewardTerm = term(RewardTerm, func=mdp.joint_pos_limits, weight=-1.0)
  action_rate_l2: RewardTerm = term(RewardTerm, func=mdp.action_rate_l2, weight=-0.1)

  # Rewards feet being airborne for 0.05-0.5 seconds.
  # Lift your feet off the ground and keep them up for a reasonable amount of time.
  air_time: RewardTerm = term(
    RewardTerm,
    func=mdp.feet_air_time,
    weight=0.0,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.05,
      "threshold_max": 0.5,
      "command_name": "twist",
      "command_threshold": 0.5,
    },
  )
  # Guide the foot height during the swing phase.
  # Large penalty when foot is moving fast and far from target height.
  # This is a dense reward.
  foot_clearance: RewardTerm = term(
    RewardTerm,
    func=mdp.feet_clearance,
    weight=-2.0,
    params={
      "target_height": 0.1,
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", geom_names=[]),
    },
  )
  # Tracks peak height during swing. Did you actually reach 0.1m at some point?
  # This is a sparse reward, only evaluated at landing.
  foot_swing_height: RewardTerm = term(
    RewardTerm,
    func=mdp.feet_swing_height,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "target_height": 0.1,
      "command_name": "twist",
      "command_threshold": 0.05,
      "num_feet": 4,
      "asset_cfg": SceneEntityCfg("robot", geom_names=[]),  # Override in robot cfg.
    },
  )
  # Don't slide when foot is on ground.
  foot_slip: RewardTerm = term(
    RewardTerm,
    func=mdp.feet_slip,
    weight=-0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", geom_names=[]),  # Override in robot cfg.
    },
  )


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  fell_over: DoneTerm = term(
    DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
  )
  illegal_contact: DoneTerm | None = term(
    DoneTerm,
    func=mdp.illegal_contact,
    params={"sensor_name": "nonfoot_ground_touch"},
  )


@dataclass
class CurriculumCfg:
  terrain_levels: CurrTerm | None = term(
    CurrTerm, func=mdp.terrain_levels_vel, params={"command_name": "twist"}
  )


##
# Environment.
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


@dataclass
class LocomotionVelocityEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)
  curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
  decimation: int = 4  # 50 Hz control frequency.
  episode_length_s: float = 20.0

  def __post_init__(self):
    # Enable curriculum mode for terrain generator.
    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = True
