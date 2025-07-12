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
from mjlab.envs.mdp import rewards, observations, actions
from mjlab.envs.manager_based_env import ManagerBasedEnv, ManagerBasedEnvCfg


##
# Scene.
##


SCENE_CFG = SceneCfg(
  terrains={"floor": FLAT_TERRAIN_CFG},
  robots={"g1": G1_ROBOT_CFG, "g2": G1_ROBOT_CFG},
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
      params={"entity_cfg": SceneEntityCfg("g2", joint_names=[".*ankle"])},
    )

    hip_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos,
      params={"entity_cfg": SceneEntityCfg("g2", joint_names=[".*hip"])},
    )

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class CriticCfg(ObsGroup):
    ankle_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos,
      params={"entity_cfg": SceneEntityCfg("g2", joint_names=[".*ankle"])},
    )

    hip_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos,
      params={"entity_cfg": SceneEntityCfg("g2", joint_names=[".*hip"])},
    )

    waist_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos,
      params={"entity_cfg": SceneEntityCfg("g2", joint_names=[".*waist"])},
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
  pass


# Curriculum.


@dataclass
class CurriculumCfg:
  pass


# Actions.


@dataclass
class ActionCfg:
  joint_pos: ActionTerm = term(
    actions.JointPositionActionCfg,
    asset_name="g2",
    joint_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


##
# Environment.
##

# Put everything together.


@dataclass
class Go1LocomotionFlatEnvCfg(ManagerBasedEnvCfg):
  decimation: int = 1
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  # commands: CommandsCfg = CommandsCfg()
  # events: EventCfg = EventCfg()
  # rewards: RewardCfg = RewardCfg()
  # terminations: TerminationCfg = TerminationCfg()
  # curriculum: CurriculumCfg = CurriculumCfg()


if __name__ == "__main__":
  env_cfg = Go1LocomotionFlatEnvCfg()
  env = ManagerBasedEnv(cfg=env_cfg)
