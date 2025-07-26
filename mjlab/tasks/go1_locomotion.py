from dataclasses import dataclass, field

from mjlab.scene.scene_config import SceneCfg
from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_ROBOT_CFG
from mjlab.asset_zoo.terrains.flat_terrain import FLAT_TERRAIN_CFG
from mjlab.utils.spec_editor.spec_editor_config import TextureCfg, LightCfg

from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import ActionTermCfg as ActionTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import (
  observations,
  actions,
  terminations,
  events,
  commands,
  rewards,
)
from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.sensors import ContactSensorCfg
import math

from mjlab.tasks.mdp import rewards as custom_rewards
from mjlab.utils.noise import UniformNoiseCfg as Unoise


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
  robots={"robot": GO1_ROBOT_CFG},
  sensors={
    "contact_forces": ContactSensorCfg(
      entity_name="robot",
      history_length=0,
      filter_expr=[".*"],
    ),
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
    actions.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=0.25,
    use_default_offset=True,
  )


# Commands.


@dataclass
class CommandsCfg:
  base_velocity: commands.UniformVelocityCommandCfg = term(
    commands.UniformVelocityCommandCfg,
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.02,
    rel_heading_envs=1.0,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=commands.UniformVelocityCommandCfg.Ranges(
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
      func=observations.base_lin_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=observations.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=observations.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=observations.joint_vel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    actions: ObsTerm = term(
      ObsTerm,
      func=observations.last_action,
    )
    velocity_commands: ObsTerm = term(
      ObsTerm,
      func=observations.generated_commands,
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
    func=events.reset_root_state_uniform,
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
    func=events.reset_joints_by_scale,
    mode="reset",
    params={
      "position_range": (1.0, 1.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    },
  )


# Rewards.


@dataclass
class RewardCfg:
  # Task.
  track_lin_vel_xy_exp: RewardTerm = term(
    RewardTerm,
    func=rewards.track_lin_vel_xy_exp,
    weight=1.5,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
  )
  track_ang_vel_z_exp: RewardTerm = term(
    RewardTerm,
    func=rewards.track_ang_vel_z_exp,
    weight=0.75,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
  )
  # Penalties.
  lin_vel_z_l2: RewardTerm = term(RewardTerm, func=rewards.lin_vel_z_l2, weight=-2.0)
  ang_vel_xy_l2: RewardTerm = term(RewardTerm, func=rewards.ang_vel_xy_l2, weight=-0.05)
  dof_torques_l2: RewardTerm = term(
    RewardTerm,
    func=rewards.joint_torques_l2,
    weight=-0.0002,
  )
  dof_acc_l2: RewardTerm = term(RewardTerm, func=rewards.joint_acc_l2, weight=-2.5e-7)
  action_rate_l2: RewardTerm = term(
    RewardTerm, func=rewards.action_rate_l2, weight=-0.01
  )
  flat_orientation_l2: RewardTerm = term(
    RewardTerm, func=rewards.flat_orientation_l2, weight=-2.5
  )
  dof_pos_limits: RewardTerm = term(
    RewardTerm, func=rewards.joint_pos_limits, weight=0.0
  )
  feet_air_time: RewardTerm = term(
    RewardTerm,
    func=custom_rewards.feet_air_time,
    weight=0.25,
    params={
      "sensor_cfg": SceneEntityCfg("feet_contact_forces"),
      "command_name": "base_velocity",
      "threshold": 0.5,
    },
  )
  # undesired_contacts: RewardTerm = term(
  #   RewardTerm,
  #   func=rewards.undesired_contacts,
  #   weight=-1.0,
  #   params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"), "threshold": 1.0},
  # )


# Terminations.


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=terminations.time_out, time_out=True)
  base_contact: DoneTerm = term(
    DoneTerm,
    func=terminations.illegal_contact,
    params={
      "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
      "threshold": 1.0,
    },
  )


# Curriculum.


@dataclass
class CurriculumCfg:
  pass


##
# Environment.
##

# Put everything together.


@dataclass
class Go1LocomotionFlatEnvCfg(ManagerBasedRlEnvCfg):
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
    self.sim.mujoco.iterations = 10
    self.sim.mujoco.ls_iterations = 20
    self.sim.ls_parallel = False
    self.sim.num_envs = 1
    self.sim.nconmax = 32768
    self.sim.njmax = 75


# if __name__ == "__main__":
#   from mjlab.envs import ManagerBasedRLEnv
#   import torch
#   import mujoco_warp as mjwarp
#   import mujoco
#   from mjlab.scene import Scene

#   env_cfg = Go1LocomotionFlatEnvCfg()

#   # scn = Scene(env_cfg.scene)

#   # mjm = scn.compile()
#   # mjd = mujoco.MjData(mjm)
#   # mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
#   # mujoco.mj_forward(mjm, mjd)

#   # wp_model = mjwarp.put_model(mjm)

#   # from ipdb import set_trace; set_trace()

#   # self._wp_data = mjwarp.put_data(
#   #   self._mj_model,
#   #   self._mj_data,
#   #   nworld=self.cfg.num_envs,
#   #   nconmax=self.cfg.nconmax,
#   #   njmax=self.cfg.njmax,
#   # )

#   env = ManagerBasedRLEnv(cfg=env_cfg)

#   obs, extras = env.reset()

#   action = torch.zeros((env.num_envs, env.action_manager.total_action_dim))
#   ret = env.step(action)
