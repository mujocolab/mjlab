from dataclasses import dataclass, field

from mjlab.entities.robots.robot import RobotCfg
from mjlab.entities.terrains.terrain import TerrainCfg
from mjlab.utils.spec_editor.spec_editor_config import LightCfg, CameraCfg, TextureCfg


@dataclass
class SceneCfg:
  terrains: dict[str, TerrainCfg] = field(default_factory=dict)
  robots: dict[str, RobotCfg] = field(default_factory=dict)
  lights: tuple[LightCfg, ...] = ()
  cameras: tuple[CameraCfg, ...] = ()
  skybox: TextureCfg | None = None
