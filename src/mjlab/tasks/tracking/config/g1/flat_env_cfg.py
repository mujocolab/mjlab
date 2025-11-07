from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@dataclass
class G1FlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    self.scene.entities = {"robot": replace(G1_ROBOT_CFG)}

    self_collision_cfg = ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    )
    self.scene.sensors = (self_collision_cfg,)

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

    self.commands.motion.sampling_mode = "start"

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)


@dataclass
class G1FlatEnvCfg_DEMO(G1FlatEnvCfg_PLAY):
  def __post_init__(self):
    super().__post_init__()

    # The demo uses a long motion, so we use uniform sampling to see more diversity
    # with num_envs > 1.
    self.commands.motion.sampling_mode = "uniform"


@dataclass
class G1FlatNoStateEstimationEnvCfg_PLAY(G1FlatNoStateEstimationEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    self.commands.motion.sampling_mode = "start"

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
