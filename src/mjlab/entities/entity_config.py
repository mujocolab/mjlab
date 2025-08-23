from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco

from mjlab.utils.spec_editor.spec_editor_config import (
  CameraCfg,
  LightCfg,
  MaterialCfg,
  SensorCfg,
  TextureCfg,
)


@dataclass
class EntityCfg:
  @dataclass
  class InitialStateCfg:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

  spec_fn: Callable[[], mujoco.MjSpec] = field(
    default_factory=lambda: (lambda: mujoco.MjSpec())
  )
  init_state: InitialStateCfg = field(default_factory=InitialStateCfg)
  debug_vis: bool = False

  lights: tuple[LightCfg, ...] = field(default_factory=tuple)
  cameras: tuple[CameraCfg, ...] = field(default_factory=tuple)
  textures: tuple[TextureCfg, ...] = field(default_factory=tuple)
  materials: tuple[MaterialCfg, ...] = field(default_factory=tuple)
  sensors: tuple[SensorCfg, ...] = field(default_factory=tuple)
