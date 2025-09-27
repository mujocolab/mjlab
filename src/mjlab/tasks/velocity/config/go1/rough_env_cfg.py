from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  GO1_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  create_locomotion_velocity_env_cfg,
)
from mjlab.utils.spec_config import ContactSensorCfg


def create_unitree_go1_rough_env_cfg():
  """Create configuration for Unitree GO1 robot on rough terrain."""
  # Configure foot contact sensors
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

  # Create configuration with all GO1-specific parameters
  cfg = create_locomotion_velocity_env_cfg(
    robot_cfg=go1_cfg,
    action_scale=GO1_ACTION_SCALE,
    viewer_body_name="trunk",
    viewer_distance=1.5,
    viewer_elevation=-10.0,
    air_time_sensor_names=[
      "FR_foot_ground_contact",
      "FL_foot_ground_contact",
      "RR_foot_ground_contact",
      "RL_foot_ground_contact",
    ],
    foot_friction_geom_names=[r"^(RR|RL|FR|FL)_foot_collision$"],
    foot_clearance_geom_names=[r"^(RR|RL|FR|FL)_foot_collision$"],
    pose_l2_std={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    },
  )

  return cfg


def create_unitree_go1_rough_env_cfg_play():
  """Create play configuration for Unitree GO1 robot on rough terrain."""
  cfg = create_unitree_go1_rough_env_cfg()

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
UNITREE_GO1_ROUGH_ENV_CFG = create_unitree_go1_rough_env_cfg()
UNITREE_GO1_ROUGH_ENV_CFG_PLAY = create_unitree_go1_rough_env_cfg_play()
