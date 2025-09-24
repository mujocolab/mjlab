from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.utils.spec_editor.spec_editor_config import SensorCfg


@dataclass
class UnitreeG1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.commands.twist.viz.z_offset = 0.75

    angmom_sensor = SensorCfg(
      name="root_angmom",
      sensor_type="subtreeangmom",
      kwargs={
        "objtype": "body",
        "objname": "pelvis",
      },
    )
    g1_cfg = replace(G1_ROBOT_CFG, sensors=(angmom_sensor,))

    self.scene.entities = {"robot": g1_cfg}
    self.actions.joint_pos.scale = G1_ACTION_SCALE

    self.events.foot_friction.params["asset_cfg"].geom_names = [
      r"^(left|right)_foot[1-7]_collision$"
    ]

    self.rewards.pose_l2.params["std"] = {
      r"^(left|right)_knee_joint$": 5.0,
      r"^(left|right)_hip_pitch_joint$": 5.0,
      r"^(left|right)_elbow_joint$": 5.0,
      r"^(left|right)_shoulder_pitch_joint$": 5.0,
      r"^(?!.*(knee_joint|hip_pitch|elbow_joint|shoulder_pitch)).*$": 0.3,
    }

    self.viewer.body_name = "torso_link"


@dataclass
class UnitreeG1RoughEnvCfg_PLAY(UnitreeG1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0
