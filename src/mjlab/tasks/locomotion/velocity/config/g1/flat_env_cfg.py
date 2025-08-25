from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ROBOT_CFG
from mjlab.sensors import ContactSensorCfg
from mjlab.tasks.locomotion.velocity.velocity_env_cfg import (
  LocomotionVelocityFlatEnvCfg,
)


@dataclass
class UnitreeG1FlatEnvCfg(LocomotionVelocityFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    g1_cfg = replace(G1_ROBOT_CFG)
    g1_cfg.joint_pos_weight = {".*knee_joint": 0.1}
    self.scene.robots = {"robot": g1_cfg}
    self.events.push_robot = None
    self.observations.policy.enable_corruption = False
    self.scene.sensors = {
      "feet_contact_forces": ContactSensorCfg(
        entity_name="robot",
        history_length=3,
        track_air_time=True,
        filter_expr=[".*knee.*"],
        geom_filter_expr=[".*_foot3_collision"],
      ),
    }
    self.sim.njmax = 300
    self.sim.nconmax = 100000
