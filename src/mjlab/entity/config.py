from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco

from mjlab.utils.spec_editor.spec_editor_config import (
  ActuatorCfg,
  CameraCfg,
  CollisionCfg,
  LightCfg,
  MaterialCfg,
  SensorCfg,
  TextureCfg,
)


@dataclass
class EntityCfg:
  @dataclass
  class InitialStateCfg:
    # Freejoint.
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Articulation.
    joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})

  init_state: InitialStateCfg = field(default_factory=InitialStateCfg)
  spec_fn: Callable[[], mujoco.MjSpec] = field(
    default_factory=lambda: (lambda: mujoco.MjSpec())
  )
  articulation: EntityArticulationInfoCfg | None = None

  # Editors.
  lights: tuple[LightCfg, ...] = field(default_factory=tuple)
  cameras: tuple[CameraCfg, ...] = field(default_factory=tuple)
  textures: tuple[TextureCfg, ...] = field(default_factory=tuple)
  materials: tuple[MaterialCfg, ...] = field(default_factory=tuple)
  sensors: tuple[SensorCfg, ...] = field(default_factory=tuple)
  collisions: tuple[CollisionCfg, ...] = field(default_factory=tuple)

  # Misc.
  debug_vis: bool = False


@dataclass
class EntityArticulationInfoCfg:
  actuators: tuple[ActuatorCfg, ...] = field(default_factory=tuple)
  soft_joint_pos_limit_factor: float = 1.0
  joint_pos_weight: dict[str, float] | None = None
