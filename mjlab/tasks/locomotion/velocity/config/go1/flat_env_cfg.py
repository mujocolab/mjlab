from dataclasses import dataclass

from mjlab.tasks.locomotion.velocity.velocity_env_cfg import (
  LocomotionVelocityFlatEnvCfg,
)
from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_ROBOT_CFG


@dataclass
class UnitreeGo1FlatEnvCfg(LocomotionVelocityFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.robots = {"robot": GO1_ROBOT_CFG}
    self.rewards.undesired_contacts = None
