from dataclasses import dataclass

from mjlab.sensors.sensor_base import SensorBase


@dataclass(kw_only=True)
class SensorBaseCfg:
  entity_name: str
  class_type: type[SensorBase]
  update_period: float = 0.0
  history_length: int = 0
