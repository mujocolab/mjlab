from dataclasses import dataclass, field

from mjlab.scene.scene_config import SceneCfg
from mjlab.asset_zoo.terrains.flat_terrain import FLAT_TERRAIN_CFG
from mjlab.utils.spec_editor.spec_editor_config import TextureCfg, LightCfg

from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import ActionTermCfg as ActionTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.sensors import ContactSensorCfg
import math
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.tasks.locomotion.velocity import mdp


##
# Scene.
##

terrain_cfg = FLAT_TERRAIN_CFG
terrain_cfg.textures.append(
  TextureCfg(
    name="skybox",
    type="skybox",
    builtin="gradient",
    rgb1=(0.3, 0.5, 0.7),
    rgb2=(0.1, 0.2, 0.3),
    width=512,
    height=3072,
  ),
)
terrain_cfg.lights.append(
  LightCfg(pos=(0, 0, 1.5), type="directional"),
)

SCENE_CFG = SceneCfg(
  terrains={"floor": FLAT_TERRAIN_CFG},
  sensors={
    "feet_contact_forces": ContactSensorCfg(
      entity_name="robot",
      history_length=3,
      track_air_time=True,
      filter_expr=[".*calf"],
      geom_filter_expr=[".*_foot_collision*"],
    ),
  },
)

##
# MDP.
##

# Actions.


@dataclass
class ActionCfg:
  joint_pos: ActionTerm = term(
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
      func=mdp.joint_vel,
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
      self.concatenate_terms = True

  policy: PolicyCfg = field(default_factory=PolicyCfg)


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
  push_robot: EventTerm = term(
    EventTerm,
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(10.0, 15.0),
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
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
  lin_vel_z_l2: RewardTerm = term(RewardTerm, func=mdp.lin_vel_z_l2, weight=-0.5)
  ang_vel_xy_l2: RewardTerm = term(RewardTerm, func=mdp.ang_vel_xy_l2, weight=-0.05)
  dof_torques_l2: RewardTerm = term(
    RewardTerm, func=mdp.joint_torques_l2, weight=-0.0002
  )
  dof_acc_l2: RewardTerm = term(RewardTerm, func=mdp.joint_acc_l2, weight=0.0)
  action_rate_l2: RewardTerm = term(RewardTerm, func=mdp.action_rate_l2, weight=-0.01)
  flat_orientation_l2: RewardTerm = term(
    RewardTerm, func=mdp.flat_orientation_l2, weight=-5.0
  )
  dof_pos_limits: RewardTerm = term(RewardTerm, func=mdp.joint_pos_limits, weight=-1.0)
  feet_air_time: RewardTerm = term(
    RewardTerm,
    func=mdp.feet_air_time,
    weight=0.1,
    params={
      "sensor_cfg": SceneEntityCfg("feet_contact_forces"),
      "command_name": "base_velocity",
      "threshold": 0.1,
    },
  )


# Terminations.


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  fell_over: DoneTerm = term(
    DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(45.0)}
  )


##
# Environment.
##


@dataclass
class LocomotionVelocityFlatEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  decimation: int = 4
  rewards: RewardCfg = field(default_factory=RewardCfg)
  episode_length_s: float = 20.0
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)

  def __post_init__(self):
    self.sim.mujoco.integrator = "implicitfast"
    self.sim.mujoco.cone = "pyramidal"
    self.sim.mujoco.timestep = 0.005
    self.sim.num_envs = 4096
    self.sim.nconmax = 40000
    self.sim.njmax = 100
    self.sim.mujoco.iterations = 10
    self.sim.mujoco.ls_iterations = 20
