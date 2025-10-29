from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  GO1_ROBOT_CFG,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)


@dataclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.entities = {"robot": replace(GO1_ROBOT_CFG)}

    foot_names = ["FR", "FL", "RR", "RL"]
    geom_names = [f"{name}_foot_collision" for name in foot_names]

    feet_ground_cfg = ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
      name="nonfoot_ground_touch",
      primary=ContactMatch(
        mode="geom",
        entity="robot",
        # Grab all collision geoms...
        pattern=r".*_collision\d*$",
        # Except for the foot geoms.
        exclude=tuple(geom_names),
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    )
    self.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)

    self.actions.joint_pos.scale = GO1_ACTION_SCALE

    foot_names = ["FR", "FL", "RR", "RL"]
    geom_names = [f"{name}_foot_collision" for name in foot_names]
    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    self.rewards.pose.params["std_standing"] = {
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
    }
    self.rewards.pose.params["std_moving"] = {
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    }
    self.rewards.foot_clearance.params["asset_cfg"].geom_names = geom_names
    self.rewards.foot_swing_height.params["asset_cfg"].geom_names = geom_names
    self.rewards.foot_slip.params["asset_cfg"].geom_names = geom_names

    self.observations.critic.foot_height.params["asset_cfg"].geom_names = geom_names

    self.viewer.body_name = "trunk"
    self.viewer.distance = 1.5
    self.viewer.elevation = -10.0


@dataclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
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
