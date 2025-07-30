from dataclasses import dataclass, field

from mjlab.sensors.sensor_base_config import SensorBaseCfg
from mjlab.sensors.contact_sensor.contact_sensor import ContactSensor


@dataclass
class ContactSensorCfg(SensorBaseCfg):
  class_type: type = ContactSensor
  track_air_time: bool = False
  force_threshold: float = 1.0
  filter_expr: list[str] = field(default_factory=list)
  # TODO: This is gross, figure out a better way to do this.
  geom_filter_expr: list[str] = field(default_factory=lambda: [".*"])
