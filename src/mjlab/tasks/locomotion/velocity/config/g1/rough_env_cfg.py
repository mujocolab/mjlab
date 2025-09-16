from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ROBOT_CFG,
)
from mjlab.tasks.locomotion.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)


@dataclass
class UnitreeG1EnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.commands.base_velocity.viz.z_offset = 0.75

    self.scene.entities = {"robot": replace(G1_ROBOT_CFG)}
    self.actions.joint_pos.scale = G1_ACTION_SCALE

    self.events.foot_friction.params["asset_cfg"].geom_names = [
      r"^(left|right)_foot[1-7]_collision$"
    ]

    self.rewards.pose_l2.params["std"] = {
      r"^(left|right)_knee_joint$": 1.2,
      r"^(left|right)_hip_pitch_joint$": 1.2,
      r"^(?!.*(knee_joint|hip_pitch)).*$": 0.3,
    }


@dataclass
class UnitreeG1EnvCfg_PLAY(UnitreeG1EnvCfg):
  def __post_init__(self):
    super().__post_init__()

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0
