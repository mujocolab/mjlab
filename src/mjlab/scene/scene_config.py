from dataclasses import dataclass, field

from mjlab.entities.robots.robot import RobotCfg
from mjlab.entities.terrains.terrain import TerrainCfg
from mjlab.sensors.sensor_base_config import SensorBaseCfg


@dataclass
class SceneCfg:
  """Configuration for the scene."""

  terrains: dict[str, TerrainCfg] = field(default_factory=dict)
  robots: dict[str, RobotCfg] = field(default_factory=dict)
  sensors: dict[str, SensorBaseCfg] = field(default_factory=dict)
  lazy_sensor_update: bool = True
  """Whether to update sensors only when they are accessed. Default is True."""
