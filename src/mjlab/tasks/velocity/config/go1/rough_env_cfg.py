from dataclasses import dataclass

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  GO1_ROBOT_CFG,
)
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg


@dataclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.entities = {"robot": GO1_ROBOT_CFG}

    # One sensor for all feet using regex pattern matching.
    self.scene.sensors = (
      ContactSensorCfg(
        name="feet_ground_contact",
        entity_name="robot",
        geom1=r"(FR|FL|RR|RL)_foot_collision",  # Regex matches all 4 feet
        body2="terrain",
        secondary_entity="",
        track_air_time=True,  # Enable air time tracking for locomotion rewards
        data=("found", "force"),
        reduce="netforce",
      ),
    )

    self.actions.joint_pos.scale = GO1_ACTION_SCALE

    # Update rewards to use the single sensor.
    self.rewards.air_time.params["sensor_name"] = "feet_ground_contact"
    geom_names = [f"{name}_foot_collision" for name in ["FR", "FL", "RR", "RL"]]
    self.rewards.pose.params["std"] = {
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    }

    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

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

    self.curriculum.command_vel = None
    self.commands.twist.ranges.lin_vel_x = (-3.0, 3.0)
    self.commands.twist.ranges.ang_vel_z = (-3.0, 3.0)
