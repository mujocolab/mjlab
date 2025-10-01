from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ROBOT_CFG,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from mjlab.utils.spec_config import ContactSensorCfg


def create_g1_rough_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create configuration for Unitree G1 robot velocity tracking on rough terrain."""
  # Configure foot contact sensors.
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

  sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
  geom_names = []
  for i in range(1, 8):
    geom_names.append(f"left_foot{i}_collision")
  for i in range(1, 8):
    geom_names.append(f"right_foot{i}_collision")

  posture_std_dict = {
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

  # Create the base velocity config.
  cfg = create_velocity_env_cfg(
    robot_cfg=g1_cfg,
    action_scale=G1_ACTION_SCALE,
    viewer_body_name="torso_link",
    foot_friction_geom_names=geom_names,
    feet_sensor_names=sensor_names,
    posture_std=[posture_std_dict],
  )

  # Disable command velocity curriculum (set in base config).
  assert cfg.curriculum is not None
  assert "command_vel" in cfg.curriculum
  del cfg.curriculum["command_vel"]

  # Update twist command visualization.
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  assert cfg.commands is not None
  assert "twist" in cfg.commands
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.75

  return cfg


def create_g1_rough_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create play configuration for Unitree G1 robot velocity tracking on rough terrain."""
  cfg = create_g1_rough_env_cfg()

  # Effectively infinite episode length.
  cfg.episode_length_s = int(1e9)

  # Disable curriculum for terrain generator.
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_generator.curriculum = False
  cfg.scene.terrain.terrain_generator.num_cols = 5
  cfg.scene.terrain.terrain_generator.num_rows = 5
  cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


UNITREE_G1_ROUGH_ENV_CFG = create_g1_rough_env_cfg()
UNITREE_G1_ROUGH_ENV_CFG_PLAY = create_g1_rough_env_cfg_play()
