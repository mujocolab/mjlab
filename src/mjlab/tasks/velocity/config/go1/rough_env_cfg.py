# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  GO1_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.utils.spec_config import ContactSensorCfg


@dataclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    foot_contact_sensors = [
      ContactSensorCfg(
        name=f"{leg}_foot_ground_contact",
        geom1=f"{leg}_foot_collision",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      )
      for leg in ["FR", "FL", "RR", "RL"]
    ]
    go1_cfg = replace(GO1_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
    self.scene.entities = {"robot": go1_cfg}

    self.actions.joint_pos.scale = GO1_ACTION_SCALE

    foot_names = ["FR", "FL", "RR", "RL"]
    sensor_names = [f"{name}_foot_ground_contact" for name in foot_names]
    geom_names = [f"{name}_foot_collision" for name in foot_names]

    self.rewards.air_time.params["sensor_names"] = sensor_names
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
