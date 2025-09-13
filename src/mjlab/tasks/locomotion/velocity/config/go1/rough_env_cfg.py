from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  GO1_ROBOT_CFG,
)
from mjlab.tasks.locomotion.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)


@dataclass
class UnitreeGo1EnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.entities = {"robot": replace(GO1_ROBOT_CFG)}
    self.actions.joint_pos.scale = GO1_ACTION_SCALE


@dataclass
class UnitreeGo1EnvCfg_PLAY(UnitreeGo1EnvCfg):
  def __post_init__(self):
    super().__post_init__()

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
