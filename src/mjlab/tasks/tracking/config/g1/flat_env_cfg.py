from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@dataclass
class G1FlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.robots = {"robot": replace(G1_ROBOT_CFG)}
    self.actions.joint_pos.scale = G1_ACTION_SCALE

    self.commands.motion.reference_body = "torso_link"
    self.commands.motion.body_names = [
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "torso_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    ]


@dataclass
class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False

    self.events.push_robot = None

    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}
    self.commands.motion.start_from_beginning = True

    self.episode_length_s = int(1e9)  # effectively infinite episode length
