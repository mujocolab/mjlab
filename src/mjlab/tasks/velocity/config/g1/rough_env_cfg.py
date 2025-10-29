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
    self.scene.sensors = (feet_ground_cfg,)

    self.actions.joint_pos.scale = G1_ACTION_SCALE

    # Use all foot geoms for friction randomization
    geom_names = []
    for i in range(1, 8):
      geom_names.append(f"left_foot{i}_collision")
    for i in range(1, 8):
      geom_names.append(f"right_foot{i}_collision")
    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    # self.rewards.pose.params["std"] = {
    #   # Lower body.
    #   r".*hip_pitch.*": 0.3,
    #   r".*hip_roll.*": 0.15,
    #   r".*hip_yaw.*": 0.15,
    #   r".*knee.*": 0.35,
    #   r".*ankle_pitch.*": 0.25,
    #   r".*ankle_roll.*": 0.1,
    #   # Waist.
    #   r".*waist_yaw.*": 0.15,
    #   r".*waist_roll.*": 0.08,
    #   r".*waist_pitch.*": 0.1,
    #   # Arms.
    #   r".*shoulder_pitch.*": 0.35,
    #   r".*shoulder_roll.*": 0.15,
    #   r".*shoulder_yaw.*": 0.1,
    #   r".*elbow.*": 0.25,
    #   r".*wrist.*": 0.3,
    # }
    self.rewards.pose.params["std_standing"] = {
      # Lower body.
      r".*hip_pitch.*": 0.05,  # Very tight!
      r".*hip_roll.*": 0.05,
      r".*hip_yaw.*": 0.05,
      r".*knee.*": 0.1,  # Tight!
      r".*ankle_pitch.*": 0.05,
      r".*ankle_roll.*": 0.05,
      # Waist.
      r".*waist_yaw.*": 0.05,
      r".*waist_roll.*": 0.05,
      r".*waist_pitch.*": 0.05,
      # Arms.
      r".*shoulder_pitch.*": 0.1,
      r".*shoulder_roll.*": 0.05,
      r".*shoulder_yaw.*": 0.05,
      r".*elbow.*": 0.05,
      r".*wrist.*": 0.1,
    }
    self.rewards.pose.params["std_moving"] = {
      # Lower body.
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      # Waist.
      r".*waist_yaw.*": 0.15,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.1,
      # Arms.
      r".*shoulder_pitch.*": 0.35,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.25,
      r".*wrist.*": 0.3,
    }
    self.rewards.foot_clearance.params["asset_cfg"].geom_names = geom_names
    self.rewards.foot_swing_height.params["asset_cfg"].geom_names = geom_names
    self.rewards.foot_slip.params["asset_cfg"].geom_names = geom_names
    self.rewards.foot_swing_height.params["num_feet"] = 2

    self.observations.critic.foot_height.params["asset_cfg"].geom_names = geom_names

    self.terminations.illegal_contact = None

    self.viewer.body_name = "torso_link"
    self.commands.twist.viz.z_offset = 0.75


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
