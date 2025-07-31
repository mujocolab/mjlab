from dataclasses import dataclass, replace

from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ROBOT_CFG


@dataclass
class G1FlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.robots = {"robot": replace(G1_ROBOT_CFG)}
