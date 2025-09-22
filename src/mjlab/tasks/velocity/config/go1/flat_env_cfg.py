from dataclasses import dataclass

from mjlab.tasks.velocity.config.go1.rough_env_cfg import (
  UnitreeGo1RoughEnvCfg,
)


@dataclass
class UnitreeGo1FlatEnvCfg(UnitreeGo1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None


@dataclass
class UnitreeGo1FlatEnvCfg_PLAY(UnitreeGo1FlatEnvCfg):
  pass
