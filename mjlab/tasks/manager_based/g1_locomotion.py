from dataclasses import dataclass, field

from mjlab.entities.scene.scene_config import SceneCfg, LightCfg
from mjlab.entities.common.config import TextureCfg
from mjlab.entities.robots.g1.g1_constants import G1_ROBOT_CFG
from mjlab.entities.terrains.flat_terrain import FLAT_TERRAIN_CFG

from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import ActionTermCfg as ActionTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import rewards, observations, actions, terminations
from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv
from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm


##
# Scene.
##


SCENE_CFG = SceneCfg(
  terrains={"floor": FLAT_TERRAIN_CFG},
  robots={"g1": G1_ROBOT_CFG},
  lights=(LightCfg(pos=(0, 0, 1.5), type="directional"),),
  skybox=TextureCfg(
    name="skybox",
    type="skybox",
    builtin="gradient",
    rgb1=(0.3, 0.5, 0.7),
    rgb2=(0.1, 0.2, 0.3),
    width=512,
    height=3072,
  ),
)

##
# MDP.
##

# Commands.


@dataclass
class CommandsCfg:
  pass


# Observations.


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    ankle_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos,
      params={"entity_cfg": SceneEntityCfg("g1", joint_names=[".*ankle"])},
    )

    hip_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos,
      params={"entity_cfg": SceneEntityCfg("g1", joint_names=[".*hip"])},
    )

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class CriticCfg(ObsGroup):
    ankle_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos,
      params={"entity_cfg": SceneEntityCfg("g1", joint_names=[".*ankle"])},
    )

    hip_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos,
      params={"entity_cfg": SceneEntityCfg("g1", joint_names=[".*hip"])},
    )

    waist_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos,
      params={"entity_cfg": SceneEntityCfg("g1", joint_names=[".*waist"])},
    )

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: CriticCfg = field(default_factory=CriticCfg)


# Events.


@dataclass
class EventCfg:
  pass


# Rewards.


@dataclass
class RewardCfg:
  dof_torques: RewardTerm = term(
    RewardTerm,
    func=rewards.joint_torques_l2,
    weight=-1e-4,
    params={"entity_cfg": SceneEntityCfg("g1", joint_names=[".*"])},
  )


# Terminations.


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=terminations.time_out, time_out=True)


# Curriculum.


@dataclass
class CurriculumCfg:
  pass


# Actions.


@dataclass
class ActionCfg:
  joint_pos: ActionTerm = term(
    actions.JointPositionActionCfg,
    asset_name="g1",
    joint_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


##
# Environment.
##

# Put everything together.


@dataclass
class Go1LocomotionFlatEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  decimation: int = 1
  rewards: RewardCfg = field(default_factory=RewardCfg)
  episode_length_s: float = 10.0
  # commands: CommandsCfg = CommandsCfg()
  # events: EventCfg = EventCfg()
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  # curriculum: CurriculumCfg = CurriculumCfg()

  def __post_init__(self):
    self.sim.mujoco.integrator = "implicitfast"
    self.sim.mujoco.cone = "pyramidal"
    self.sim.mujoco.timestep = 0.005
    self.sim.mujoco.iterations = 10
    self.sim.mujoco.ls_iterations = 20
    self.sim.num_envs = 4096
    self.sim.nconmax = 32768


if __name__ == "__main__":
  import torch
  from tqdm.auto import tqdm

  env_cfg = Go1LocomotionFlatEnvCfg()
  env = ManagerBasedRLEnv(cfg=env_cfg)

  obs, extras = env.reset()
  for _ in tqdm(range(25)):
    action = torch.rand(
      (env.num_envs, env.action_manager.total_action_dim), device="cuda:0"
    )
    env.step(action)
