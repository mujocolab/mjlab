from dataclasses import dataclass


@dataclass(frozen=True)
class TextureCfg:
  name: str
  type: str
  builtin: str
  rgb1: tuple[float, float, float]
  rgb2: tuple[float, float, float]
  width: int
  height: int
  mark: str = "none"
  markrgb: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MaterialCfg:
  name: str
  texuniform: bool
  texrepeat: tuple[int, int]
  reflectance: float = 0.0
  texture: str | None = None


@dataclass(frozen=True)
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
