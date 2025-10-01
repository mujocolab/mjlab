from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  GO1_ROBOT_CFG,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from mjlab.utils.spec_config import ContactSensorCfg


def create_go1_rough_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create configuration for Unitree Go1 robot velocity tracking on rough terrain."""
  # Configure foot contact sensors.
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

  foot_names = ["FR", "FL", "RR", "RL"]
  sensor_names = [f"{name}_foot_ground_contact" for name in foot_names]
  geom_names = [f"{name}_foot_collision" for name in foot_names]

  posture_std_dict = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }

  # Create the base velocity config.
  cfg = create_velocity_env_cfg(
    robot_cfg=go1_cfg,
    action_scale=GO1_ACTION_SCALE,
    viewer_body_name="trunk",
    foot_friction_geom_names=geom_names,
    feet_sensor_names=sensor_names,
    posture_std=posture_std_dict,
  )

  # Update viewer settings.
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  return cfg


def create_go1_rough_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create play configuration for Unitree Go1 robot velocity tracking on rough terrain."""
  cfg = create_go1_rough_env_cfg()

  # Effectively infinite episode length.
  cfg.episode_length_s = int(1e9)

  # Disable curriculum for terrain generator
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_generator.curriculum = False
  cfg.scene.terrain.terrain_generator.num_cols = 5
  cfg.scene.terrain.terrain_generator.num_rows = 5
  cfg.scene.terrain.terrain_generator.border_width = 10.0

  # Update command velocity curriculum and ranges.
  assert cfg.curriculum is not None
  del cfg.curriculum["command_vel"]

  assert cfg.commands is not None
  assert "twist" in cfg.commands
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-3.0, 3.0)
  twist_cmd.ranges.ang_vel_z = (-3.0, 3.0)

  return cfg


UNITREE_GO1_ROUGH_ENV_CFG = create_go1_rough_env_cfg()
UNITREE_GO1_ROUGH_ENV_CFG_PLAY = create_go1_rough_env_cfg_play()
