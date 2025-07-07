from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class KeyframeCfg:
  root_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  root_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
  ctrl: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
  time: float = 0.0
  root_lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
  root_ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
  joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
  use_joint_pos_for_ctrl: bool = False


@dataclass
class ActuatorCfg:
  joint_names_expr: list[str]
  effort_limit: float
  stiffness: float
  damping: float
  frictionloss: float = 0.0
  armature: float = 0.0


@dataclass
class SensorCfg:
  sensor_type: str
  object_name: str
  object_type: str


@dataclass
class RobotCfg:
  xml_path: Path
  actuators: tuple[ActuatorCfg, ...]
  sensors: dict[str, SensorCfg]
  keyframes: dict[str, KeyframeCfg]
  soft_joint_pos_limit_factor: float = 1.0
  asset_fn: Callable[[], dict[str, bytes]] = field(default_factory=lambda: (lambda: {}))
