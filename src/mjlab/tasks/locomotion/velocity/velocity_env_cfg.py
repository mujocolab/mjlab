import math
from dataclasses import dataclass, field

from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import CurriculumTermCfg as CurrTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.tasks.locomotion.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise

##
# Scene.
##

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  ),
)

##
# MDP.
##

# Actions.


@dataclass
class ActionCfg:
  joint_pos: mdp.JointPositionActionCfg = term(
    mdp.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


# Commands.


@dataclass
class CommandsCfg:
  base_velocity: mdp.UniformVelocityCommandCfg = term(
    mdp.UniformVelocityCommandCfg,
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.02,
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


# Observations.


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    base_lin_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_lin_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_ang_vel,
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

    actions: ObsTerm = term(
      ObsTerm,
      func=mdp.last_action,
    )
    velocity_commands: ObsTerm = term(
      ObsTerm,
      func=mdp.generated_commands,
      params={"command_name": "base_velocity"},
    )

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class PrivilegedCfg(PolicyCfg):
    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)


# Events.


@dataclass
class EventCfg:
  reset_base: EventTerm = term(
    EventTerm,
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
      "velocity_range": {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
      },
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
    interval_range_s=(10.0, 15.0),
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
  )
  foot_friction: EventTerm = term(
    EventTerm,
    mode="startup",
    func=mdp.randomize_field,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", geom_names=[r"^(RR|RL|FR|FL)_foot_collision$"]
      ),
      "operation": "abs",
      "field": "geom_friction",
      "ranges": (0.3, 1.2),
    },
  )


# Rewards.


@dataclass
class RewardCfg:
  # Task.
  track_lin_vel_xy_exp: RewardTerm = term(
    RewardTerm,
    func=mdp.track_lin_vel_xy_exp,
    weight=1.0,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
  )
  track_ang_vel_z_exp: RewardTerm = term(
    RewardTerm,
    func=mdp.track_ang_vel_z_exp,
    weight=0.5,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
  )
  # Penalties.
  lin_vel_z_l2: RewardTerm = term(RewardTerm, func=mdp.lin_vel_z_l2, weight=0.0)
  ang_vel_xy_l2: RewardTerm = term(RewardTerm, func=mdp.ang_vel_xy_l2, weight=0.0)
  dof_torques_l2: RewardTerm = term(RewardTerm, func=mdp.joint_torques_l2, weight=0.0)
  dof_acc_l2: RewardTerm = term(RewardTerm, func=mdp.joint_acc_l2, weight=0.0)
  action_rate_l2: RewardTerm = term(RewardTerm, func=mdp.action_rate_l2, weight=-0.01)
  flat_orientation_l2: RewardTerm = term(
    RewardTerm, func=mdp.flat_orientation_l2, weight=0.0
  )
  dof_pos_limits: RewardTerm = term(RewardTerm, func=mdp.joint_pos_limits, weight=-1.0)
  pose_l2: RewardTerm = term(
    RewardTerm,
    func=mdp.posture,
    weight=-0.1,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
      "std": {
        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
      },
    },
  )
  power: RewardTerm = term(RewardTerm, func=mdp.electrical_power_cost, weight=-0.005)


# Terminations.


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  fell_over: DoneTerm = term(
    DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(160.0)}
  )


# Curriculum.


@dataclass
class CurriculumCfg:
  terrain_levels: CurrTerm = term(CurrTerm, func=mdp.terrain_levels_vel)


##
# Environment.
##


@dataclass
class LocomotionVelocityEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  decimation: int = 4
  rewards: RewardCfg = field(default_factory=RewardCfg)
  episode_length_s: float = 20.0
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)
  curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)

  def __post_init__(self):
    self.scene.num_envs = 1
    self.sim.nconmax = 140000
    self.sim.njmax = 300
    self.sim.mujoco.timestep = 0.005
    self.sim.mujoco.iterations = 10
    self.sim.mujoco.ls_iterations = 20

    # Enable curriculum mode for terrain generator.
    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = True
