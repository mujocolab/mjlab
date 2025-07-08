from dataclasses import dataclass

from mjlab.entities.robots.robot import RobotCfg
from mjlab.entities.terrains.terrain import TerrainCfg
from mjlab.entities.common.config import TextureCfg


@dataclass(frozen=True)
class LightCfg:
  name: str | None = None
  body: str = "world"
  mode: str = "fixed"
  target: str | None = None
  type: str = "spot"
  castshadow: bool = True
  pos: tuple[float, float, float] = (0, 0, 0)
  dir: tuple[float, float, float] = (0, 0, -1)
  cutoff: float = 45
  exponent: float = 10


@dataclass(frozen=True)
class CameraCfg:
  name: str
  body: str = "world"
  mode: str = "fixed"
  target: str | None = None
  fovy: float = 45
  pos: tuple[float, float, float] = (0, 0, 0)
  quat: tuple[float, float, float, float] = (1, 0, 0, 0)


# TODO(kevin): Think about where to add collision config.


@dataclass
class SceneCfg:
  terrains: tuple[TerrainCfg, ...] = ()
  robots: tuple[RobotCfg, ...] = ()
  lights: tuple[LightCfg, ...] = ()
  cameras: tuple[CameraCfg, ...] = ()
  skybox: TextureCfg | None = None
