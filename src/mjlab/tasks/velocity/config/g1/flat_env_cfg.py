from dataclasses import dataclass

from mjlab.tasks.velocity.config.g1.rough_env_cfg import (
  UnitreeG1RoughEnvCfg,
)


@dataclass
class UnitreeG1FlatEnvCfg(UnitreeG1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None


@dataclass
class UnitreeG1FlatEnvCfg_PLAY(UnitreeG1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
