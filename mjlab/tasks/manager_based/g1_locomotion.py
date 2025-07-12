from dataclasses import dataclass, field, fields

from mjlab.entities.scene.scene import Scene
from mjlab.entities.scene.scene_config import SceneCfg, LightCfg
from mjlab.entities.common.config import TextureCfg
from mjlab.entities.robots.g1.g1_constants import G1_ROBOT_CFG
from mjlab.entities.terrains.flat_terrain import FLAT_TERRAIN_CFG

from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.envs.mdp import rewards, observations
from mjlab.utils.dataclasses import get_terms

import mujoco


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

# robot
# terrain
# skybox

##
# MDP.
##

# Commands.


@dataclass
class CommandsCfg:
  pass


# Actions.


@dataclass
class ActionCfg:
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

  policy: PolicyCfg = field(default_factory=PolicyCfg)


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


##
# Environment.
##

# Put everything together.


@dataclass
class Go1LocomotionFlatEnvCfg:
  scene: SceneCfg = field(default_factory=SceneCfg)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  # commands: CommandsCfg = CommandsCfg()
  # actions: ActionCfg = ActionCfg()
  # events: EventCfg = EventCfg()
  # rewards: RewardCfg = RewardCfg()
  # terminations: TerminationCfg = TerminationCfg()
  # curriculum: CurriculumCfg = CurriculumCfg()

  def __post_init__(self):
    self.decimation = 4
    self.episode_length = 20.0


if __name__ == "__main__":
  # Construct the scene: terrain + robot + cameras/lights/skybox.
  scene = Scene(SCENE_CFG)
  data = mujoco.MjData(scene.model)

  mujoco.mj_resetDataKeyframe(scene.model, data, 0)
  mujoco.mj_forward(scene.model, data)

  obs_cfg = ObservationCfg()

  @dataclass
  class Env:
    data: mujoco.MjData

  for obs_name, obs_cfg in get_terms(obs_cfg.policy, ObsTerm).items():
    if "entity_cfg" in obs_cfg.params:
      obs_cfg.params["entity_cfg"].resolve(scene.model)

    env = Env(data=data)
    val = obs_cfg.func(env=env, **obs_cfg.params)
    print(f"{obs_name}: {val}")

  rew_cfg = RewardCfg()

  for rew_name, rew_cfg in get_terms(rew_cfg, RewardTerm).items():
    if "entity_cfg" in rew_cfg.params:
      rew_cfg.params["entity_cfg"].resolve(scene.model)

    env = Env(data=data)
    val = rew_cfg.func(env=env, **rew_cfg.params)
    print(f"{rew_name}: {val}")
