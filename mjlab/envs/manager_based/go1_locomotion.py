from dataclasses import dataclass

from mjlab.entities.robots.robot import Robot
from mjlab.entities.robots.go1 import go1_constants
from mjlab.entities import terrains

##
# Scene.
##


@dataclass
class SceneCfg:
  pass


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


@dataclass
class Go1LocomotionFlatEnvCfg:
  scene: SceneCfg = SceneCfg()
  observations: ObservationCfg = ObservationCfg()
  commands: CommandsCfg = CommandsCfg()
  actions: ActionCfg = ActionCfg()
  events: EventCfg = EventCfg()
  rewards: RewardCfg = RewardCfg()
  terminations: TerminationCfg = TerminationCfg()
  curriculum: CurriculumCfg = CurriculumCfg()

  def __post_init__(self):
    self.decimation = 4
    self.episode_length = 20.0
