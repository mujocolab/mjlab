from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_h1.h1_constants import H1_ROBOT_CFG
from mjlab.sensors import ContactSensorCfg
from mjlab.tasks.locomotion.velocity.velocity_env_cfg import (
  LocomotionVelocityFlatEnvCfg,
)


@dataclass
class UnitreeH1FlatEnvCfg(LocomotionVelocityFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    h1_cfg = replace(H1_ROBOT_CFG)
    h1_cfg.joint_pos_weight = {".*knee_joint": 0.1}
    self.scene.robots = {"robot": h1_cfg}
    self.events.push_robot = None
    self.observations.policy.enable_corruption = False
    self.scene.sensors = {
      "feet_contact_forces": ContactSensorCfg(
        entity_name="robot",
        history_length=3,
        track_air_time=True,
        filter_expr=[".*knee.*"],
        geom_filter_expr=[".*ankle_collision"],
      ),
    }
