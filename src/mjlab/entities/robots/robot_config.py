from dataclasses import dataclass, field
from mjlab.entities.entity_config import EntityCfg
from mjlab.utils.spec_editor.spec_editor_config import (
  CollisionCfg,
  ActuatorCfg,
)


@dataclass
class RobotCfg(EntityCfg):
  @dataclass
  class InitialStateCfg(EntityCfg.InitialStateCfg):
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})

  init_state: InitialStateCfg = field(default_factory=InitialStateCfg)  # type: ignore
  actuators: tuple[ActuatorCfg, ...] = field(default_factory=tuple)
  collisions: tuple[CollisionCfg, ...] = field(default_factory=tuple)
  soft_joint_pos_limit_factor: float = 1.0
  joint_pos_weight: dict[str, float] | None = None
