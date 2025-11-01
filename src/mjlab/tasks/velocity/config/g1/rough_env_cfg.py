from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ROBOT_CFG,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)


@dataclass
class UnitreeG1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.entities = {"robot": replace(G1_ROBOT_CFG)}

    # Constants.
    site_names = ["left_foot", "right_foot"]
    geom_names = []
    for i in range(1, 8):
      geom_names.append(f"left_foot{i}_collision")
    for i in range(1, 8):
      geom_names.append(f"right_foot{i}_collision")
    target_foot_height = 0.15

    # Sensors.
    feet_ground_cfg = ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    )
    self.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    # Actions.
    self.actions.joint_pos.scale = G1_ACTION_SCALE

    # Events.
    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    # Rewards.
    self.rewards.upright.params["asset_cfg"].body_names = ["torso_link"]
    # Tight control when stationary: maintain stable default pose.
    self.rewards.pose.params["std_standing"] = {".*": 0.05}
    # Moderate leg freedom for stepping, loose arms for natural pendulum swing.
    self.rewards.pose.params["std_walking"] = {
      # Lower body.
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      # Waist.
      r".*waist_yaw.*": 0.2,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.1,
      # Arms.
      r".*shoulder_pitch.*": 0.15,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.15,
      r".*wrist.*": 0.3,
    }
    # Maximum freedom for dynamic motion.
    self.rewards.pose.params["std_running"] = {
      # Lower body.
      r".*hip_pitch.*": 0.5,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.6,
      r".*ankle_pitch.*": 0.35,
      r".*ankle_roll.*": 0.15,
      # Waist.
      r".*waist_yaw.*": 0.3,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.2,
      # Arms.
      r".*shoulder_pitch.*": 0.5,
      r".*shoulder_roll.*": 0.2,
      r".*shoulder_yaw.*": 0.15,
      r".*elbow.*": 0.35,
      r".*wrist.*": 0.3,
    }
    self.rewards.foot_clearance.params["asset_cfg"].site_names = site_names
    self.rewards.foot_swing_height.params["asset_cfg"].site_names = site_names
    self.rewards.foot_slip.params["asset_cfg"].site_names = site_names
    self.rewards.foot_swing_height.params["target_height"] = target_foot_height
    self.rewards.foot_clearance.params["target_height"] = target_foot_height
    self.rewards.body_ang_vel.params["asset_cfg"].body_names = ["torso_link"]

    # Observations.
    self.observations.critic.foot_height.params["asset_cfg"].site_names = site_names

    # Terminations.
    self.terminations.illegal_contact = None

    self.viewer.body_name = "torso_link"
    self.commands.twist.viz.z_offset = 1.15


@dataclass
class UnitreeG1RoughEnvCfg_PLAY(UnitreeG1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0
