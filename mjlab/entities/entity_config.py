from dataclasses import dataclass, field
import mujoco
from typing import Callable
from mjlab.utils.spec_editor.spec_editor_config import (
  LightCfg,
  CameraCfg,
  TextureCfg,
  MaterialCfg,
  SensorCfg,
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

  lights: list[LightCfg] = field(default_factory=list)
  cameras: list[CameraCfg] = field(default_factory=list)
  textures: list[TextureCfg] = field(default_factory=list)
  materials: list[MaterialCfg] = field(default_factory=list)
  sensors: list[SensorCfg] = field(default_factory=list)
