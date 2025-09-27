"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.
"""

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
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


@dataclass
class CommandsCfg:
  motion: mdp.MotionCommandCfg = term(
    mdp.MotionCommandCfg,
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
    reference_body="",
    body_names=[],
  )


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
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    command: ObsTerm = term(
      ObsTerm, func=mdp.generated_commands, params={"command_name": "motion"}
    )
    motion_ref_pos_b: ObsTerm = term(
      ObsTerm,
      func=mdp.motion_ref_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    )
    motion_ref_ori_b: ObsTerm = term(
      ObsTerm,
      func=mdp.motion_ref_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    base_lin_vel: ObsTerm = term(
      ObsTerm, func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm, func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    )
    joint_pos: ObsTerm = term(
      ObsTerm, func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    )
    joint_vel: ObsTerm = term(
      ObsTerm, func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    )
    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class PrivilegedCfg(ObsGroup):
    command: ObsTerm = term(
      ObsTerm, func=mdp.generated_commands, params={"command_name": "motion"}
    )
    motion_ref_pos_b: ObsTerm = term(
      ObsTerm, func=mdp.motion_ref_pos_b, params={"command_name": "motion"}
    )
    motion_ref_ori_b: ObsTerm = term(
      ObsTerm, func=mdp.motion_ref_ori_b, params={"command_name": "motion"}
    )
    body_pos: ObsTerm = term(
      ObsTerm, func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    )
    body_ori: ObsTerm = term(
      ObsTerm, func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    )
    base_lin_vel: ObsTerm = term(ObsTerm, func=mdp.base_lin_vel)
    base_ang_vel: ObsTerm = term(ObsTerm, func=mdp.base_ang_vel)
    joint_pos: ObsTerm = term(ObsTerm, func=mdp.joint_pos_rel)
    joint_vel: ObsTerm = term(ObsTerm, func=mdp.joint_vel_rel)
    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)


@dataclass
class EventCfg:
  push_robot: EventTerm | None = term(
    EventTerm,
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={"velocity_range": VELOCITY_RANGE},
  )
  base_com: EventTerm = term(
    EventTerm,
    mode="startup",
    func=mdp.randomize_field,
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
  )
  add_joint_default_pos: EventTerm = term(
    EventTerm,
    mode="startup",
    func=mdp.randomize_field,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "add",
      "field": "qpos0",
      "ranges": (-0.01, 0.01),
    },
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
  motion_global_root_pos: RewTerm = term(
    RewTerm,
    func=mdp.motion_global_ref_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.3},
  )
  motion_global_root_ori: RewTerm = term(
    RewTerm,
    func=mdp.motion_global_ref_orientation_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.4},
  )
  motion_body_pos: RewTerm = term(
    RewTerm,
    func=mdp.motion_relative_body_position_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.3},
  )
  motion_body_ori: RewTerm = term(
    RewTerm,
    func=mdp.motion_relative_body_orientation_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.4},
  )
  motion_body_lin_vel: RewTerm = term(
    RewTerm,
    func=mdp.motion_global_body_linear_velocity_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 1.0},
  )
  motion_body_ang_vel: RewTerm = term(
    RewTerm,
    func=mdp.motion_global_body_angular_velocity_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 3.14},
  )

  action_rate_l2: RewTerm = term(RewTerm, func=mdp.action_rate_l2, weight=-1e-1)
  joint_limit: RewTerm = term(
    RewTerm,
    func=mdp.joint_pos_limits,
    weight=-10.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
  )
  self_collisions: RewTerm = term(
    RewTerm,
    func=mdp.self_collision_cost,
    weight=-10.0,
    params={"sensor_name": "self_collision"},
  )


@dataclass
class TerminationsCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  ref_pos: DoneTerm = term(
    DoneTerm,
    func=mdp.bad_ref_pos_z_only,
    params={"command_name": "motion", "threshold": 0.25},
  )
  ref_ori: DoneTerm = term(
    DoneTerm,
    func=mdp.bad_ref_ori,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "command_name": "motion",
      "threshold": 0.8,
    },
  )
  ee_body_pos: DoneTerm = term(
    DoneTerm,
    func=mdp.bad_motion_body_pos_z_only,
    params={
      "command_name": "motion",
      "threshold": 0.25,
      "body_names": [],  # Override in robot cfg.
    },
  )


SIM_CFG = SimulationCfg(
  nconmax=150_000,
  njmax=250,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)


@dataclass
class TrackingEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
  events: EventCfg = field(default_factory=EventCfg)
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
  decimation: int = 4  # 50 Hz control frequency.
  episode_length_s: float = 10.0
