from dataclasses import dataclass, field

from mjlab.entities.robots.robot import RobotCfg
from mjlab.entities.terrains.terrain import TerrainCfg


@dataclass
class SceneCfg:
  """Configuration for the scene."""

  terrains: dict[str, TerrainCfg] = field(default_factory=dict)
  robots: dict[str, RobotCfg] = field(default_factory=dict)
  # num_envs: int
