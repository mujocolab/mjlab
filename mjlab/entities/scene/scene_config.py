from dataclasses import dataclass, field

from mjlab.entities.robots.robot import RobotCfg
from mjlab.entities.terrains.terrain import TerrainCfg
from mjlab.entities.common.config import TextureCfg


@dataclass
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


@dataclass
class CameraCfg:
  name: str
  body: str = "world"
  mode: str = "fixed"
  target: str | None = None
  fovy: float = 45
  pos: tuple[float, float, float] = (0, 0, 0)
  quat: tuple[float, float, float, float] = (1, 0, 0, 0)


@dataclass
class SceneCfg:
  terrains: dict[str, TerrainCfg] = field(default_factory=dict)
  robots: dict[str, RobotCfg] = field(default_factory=dict)
  lights: tuple[LightCfg, ...] = ()
  cameras: tuple[CameraCfg, ...] = ()
  skybox: TextureCfg | None = None
