from dataclasses import dataclass, replace

from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.asset_zoo.robots.booster_t1.t1_constants import T1_ROBOT_CFG, T1_ACTION_SCALE


@dataclass
class T1FlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.robots = {"robot": replace(T1_ROBOT_CFG)}
    self.actions.joint_pos.scale = T1_ACTION_SCALE

    self.commands.motion.reference_body = "Trunk"
    self.commands.motion.body_names = [
      "Waist",
      "Hip_Roll_Left",
      "Shank_Left",
      "left_foot_link",
      "Hip_Roll_Right",
      "Shank_Right",
      "right_foot_link",
      "Trunk",
      "AL2",
      "AL3",
      # "left_wrist_yaw_link",
      "AR2",
      "AR3",
      # "right_wrist_yaw_link",
    ]


@dataclass
class T1FlatEnvCfg_PLAY(T1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None
