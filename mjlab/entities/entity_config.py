from dataclasses import dataclass, field
import mujoco
from typing import Callable
from mjlab.utils.spec_editor.spec_editor_config import (
  LightCfg,
  CameraCfg,
  TextureCfg,
  MaterialCfg,
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
  lights: tuple[LightCfg, ...] = ()
  cameras: tuple[CameraCfg, ...] = ()
  textures: tuple[TextureCfg, ...] = ()
  materials: tuple[MaterialCfg, ...] = ()
