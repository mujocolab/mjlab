from dataclasses import dataclass, MISSING

from mjlab.sensors.sensor_base import SensorBase


@dataclass
class SensorBaseCfg:
  entity_name: str
  class_type: type[SensorBase] = MISSING
  update_period: float = 0.0
  history_length: int = 0
