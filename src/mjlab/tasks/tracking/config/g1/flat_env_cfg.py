from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.utils.spec_config import ContactSensorCfg


@dataclass
class G1FlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    self_collision_sensor = ContactSensorCfg(
      name="self_collision",
      subtree1="pelvis",
      subtree2="pelvis",
      data=("found",),
      reduce="netforce",
      num=10,  # Report up to 10 contacts.
    )
    g1_cfg = replace(G1_ROBOT_CFG, sensors=(self_collision_sensor,))

    self.scene.entities = {"robot": g1_cfg}
    self.actions.joint_pos.scale = G1_ACTION_SCALE

    self.commands.motion.anchor_body_name = "torso_link"
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

    self.events.foot_friction.params["asset_cfg"].geom_names = [
      r"^(left|right)_foot[1-7]_collision$"
    ]
    self.events.base_com.params["asset_cfg"].body_names = "torso_link"

    self.terminations.ee_body_pos.params["body_names"] = [
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    ]

    self.viewer.body_name = "torso_link"


@dataclass
class G1FlatNoStateEstimationEnvCfg(G1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.motion_anchor_pos_b = None
    self.observations.policy.base_lin_vel = None


@dataclass
class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)


@dataclass
class G1FlatNoStateEstimationEnvCfg_PLAY(G1FlatNoStateEstimationEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
