from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TextureCfg:
  name: str
  type: Literal["2d", "cube", "skybox"]
  builtin: Literal["checker", "gradient", "flat", "none"]
  rgb1: tuple[float, float, float]
  rgb2: tuple[float, float, float]
  width: int
  height: int
  mark: Literal["edge", "cross", "random", "none"] = "none"
  markrgb: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class MaterialCfg:
  name: str
  texuniform: bool
  texrepeat: tuple[int, int]
  reflectance: float = 0.0
  texture: str | None = None


@dataclass
class CollisionCfg:
  geom_names_expr: list[str]
  contype: int | dict[str, int] = 1
  conaffinity: int | dict[str, int] = 1
  condim: int | dict[str, int] = 3
  priority: int | dict[str, int] = 0
  friction: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  solref: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  solimp: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  disable_other_geoms: bool = True


@dataclass(frozen=True)
class GeomCfg:
  name: str
  type: str
  size: tuple[float, ...]
  body: str = "world"
  rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1)
  material: str | None = None
  group: int = 0


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
  name: str
  sensor_type: str
  kwargs: dict[str, Any]


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
