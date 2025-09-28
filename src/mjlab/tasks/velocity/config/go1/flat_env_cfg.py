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
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    self.curriculum.command_vel = None
    self.commands.twist.ranges.lin_vel_x = (-3.0, 3.0)
    self.commands.twist.ranges.ang_vel_z = (-3.0, 3.0)
