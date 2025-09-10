from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_ROBOT_CFG
from mjlab.tasks.locomotion.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)


@dataclass
class UnitreeGo1EnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    go1_cfg = replace(GO1_ROBOT_CFG)
    assert go1_cfg.articulation is not None
    go1_cfg.articulation.joint_pos_weight = {".*calf_joint": 0.1}
    self.scene.entities = {"robot": go1_cfg}
