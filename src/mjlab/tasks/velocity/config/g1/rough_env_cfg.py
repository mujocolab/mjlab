from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  create_locomotion_velocity_env_cfg,
)
from mjlab.utils.spec_config import ContactSensorCfg


def create_unitree_g1_rough_env_cfg():
  """Create configuration for Unitree G1 robot on rough terrain."""
  # Configure foot contact sensors
  foot_contact_sensors = [
    ContactSensorCfg(
      name=f"{side}_foot_ground_contact",
      body1=f"{side}_ankle_roll_link",
      body2="terrain",
      num=1,
      data=("found",),
      reduce="netforce",
    )
    for side in ["left", "right"]
  ]
  g1_cfg = replace(G1_ROBOT_CFG, sensors=tuple(foot_contact_sensors))

  # Create configuration with all G1-specific parameters
  cfg = create_locomotion_velocity_env_cfg(
    robot_cfg=g1_cfg,
    action_scale=G1_ACTION_SCALE,
    viewer_body_name="torso_link",
    air_time_sensor_names=[
      "left_foot_ground_contact",
      "right_foot_ground_contact",
    ],
    foot_friction_geom_names=[r"^(left|right)_foot[1-7]_collision$"],
    foot_clearance_geom_names=[r"^(left|right)_foot[1-7]_collision$"],
    pose_l2_std={
      r"^(left|right)_knee_joint$": 5.0,
      r"^(left|right)_hip_pitch_joint$": 5.0,
      r"^(left|right)_elbow_joint$": 5.0,
      r"^(left|right)_shoulder_pitch_joint$": 5.0,
      r"^(?!.*(knee_joint|hip_pitch|elbow_joint|shoulder_pitch)).*$": 0.3,
    },
    command_viz_z_offset=0.75,
  )

  return cfg


def create_unitree_g1_rough_env_cfg_play():
  """Create play configuration for Unitree G1 robot on rough terrain."""
  cfg = create_unitree_g1_rough_env_cfg()

  # Rough terrain should have terrain generator configured
  assert cfg.scene.terrain is not None, "Scene terrain must be configured"
  assert cfg.scene.terrain.terrain_generator is not None, (
    "Terrain generator must be configured for rough terrain"
  )

  cfg.scene.terrain.terrain_generator.curriculum = False
  cfg.scene.terrain.terrain_generator.num_cols = 5
  cfg.scene.terrain.terrain_generator.num_rows = 5
  cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


# Create config instances
UNITREE_G1_ROUGH_ENV_CFG = create_unitree_g1_rough_env_cfg()
UNITREE_G1_ROUGH_ENV_CFG_PLAY = create_unitree_g1_rough_env_cfg_play()
