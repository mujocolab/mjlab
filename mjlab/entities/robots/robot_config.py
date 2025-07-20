from dataclasses import dataclass, field
from mjlab.entities.entity_config import EntityCfg
from mjlab.utils.spec_editor.spec_editor_config import (
  CollisionCfg,
  ActuatorCfg,
  SensorCfg,
)


@dataclass
class RobotCfg(EntityCfg):
  @dataclass
  class InitialStateCfg(EntityCfg.InitialStateCfg):
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})

  init_state: InitialStateCfg = field(default_factory=InitialStateCfg)
  actuators: tuple[ActuatorCfg, ...] = ()
  sensors: tuple[SensorCfg, ...] = ()
  collisions: tuple[CollisionCfg, ...] = ()
  soft_joint_pos_limit_factor: float = 1.0
