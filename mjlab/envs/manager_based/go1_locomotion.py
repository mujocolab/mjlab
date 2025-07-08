from dataclasses import dataclass, field

from mjlab.entities.scene.scene import Scene
from mjlab.entities.scene.scene_config import SceneCfg, LightCfg
from mjlab.entities.common.config import TextureCfg
from mjlab.entities.robots.go1.go1_constants import GO1_ROBOT_CFG
from mjlab.entities.terrains.flat_terrain import FLAT_TERRAIN_CFG


##
# Scene.
##

SCENE_CFG = SceneCfg(
  terrains=(FLAT_TERRAIN_CFG,),
  robots=(GO1_ROBOT_CFG,),
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
  pass


# Events.


@dataclass
class EventCfg:
  pass


# Rewards.


@dataclass
class RewardCfg:
  pass


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


# @dataclass
# class Go1LocomotionFlatEnvCfg:
#   scene: SceneCfg = SceneCfg()
#   observations: ObservationCfg = ObservationCfg()
#   commands: CommandsCfg = CommandsCfg()
#   actions: ActionCfg = ActionCfg()
#   events: EventCfg = EventCfg()
#   rewards: RewardCfg = RewardCfg()
#   terminations: TerminationCfg = TerminationCfg()
#   curriculum: CurriculumCfg = CurriculumCfg()

#   def __post_init__(self):
#     self.decimation = 4
#     self.episode_length = 20.0


@dataclass
class Go1LocomotionFlatEnvCfg:
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)


if __name__ == "__main__":
  import mujoco.viewer
  import tyro

  cfg = tyro.cli(Go1LocomotionFlatEnvCfg)
  scene = Scene(cfg.scene)

  # scene = Scene(SCENE_CFG)
  # scene.write_xml("test.xml")
  mujoco.viewer.launch(scene.compile())
