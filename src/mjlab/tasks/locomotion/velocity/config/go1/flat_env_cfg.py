from __future__ import annotations

from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_ROBOT_CFG
from mjlab.tasks.locomotion.velocity.velocity_env_cfg import (
  LocomotionVelocityFlatEnvCfg,
)


@dataclass
class UnitreeGo1FlatEnvCfg(LocomotionVelocityFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    go1_cfg = replace(GO1_ROBOT_CFG)
    go1_cfg.joint_pos_weight = {".*calf_joint": 0.1}
    self.scene.robots = {"robot": go1_cfg}
    self.events.push_robot = None
    self.observations.policy.enable_corruption = False
