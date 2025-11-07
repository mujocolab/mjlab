"""Unitree Go1 rough terrain velocity tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the Go1 robot on rough terrain.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_ACTION_SCALE, GO1_ROBOT_CFG
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import (
  SCENE_CFG,
  SIM_CFG,
  VIEWER_CONFIG,
  create_velocity_actions,
  create_velocity_commands,
  create_velocity_curriculum,
  create_velocity_events,
  create_velocity_observations,
  create_velocity_rewards,
  create_velocity_terminations,
)


def create_unitree_go1_rough_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain velocity tracking configuration."""
  # Go1-specific metadata.
  foot_names = ["FR", "FL", "RR", "RL"]
  site_names = ["FR", "FL", "RR", "RL"]
  geom_names = [f"{name}_foot_collision" for name in foot_names]

  # Create scene with Go1 robot and sensors.
  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": replace(GO1_ROBOT_CFG)}

  # Create sensors.
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
  scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)

  # Enable curriculum mode for terrain generator.
  if scene.terrain is not None and scene.terrain.terrain_generator is not None:
    scene.terrain.terrain_generator.curriculum = True

  # Get base configs.
  actions = create_velocity_actions()
  commands = create_velocity_commands()
  observations = create_velocity_observations()
  events = create_velocity_events()
  rewards = create_velocity_rewards()
  terminations = create_velocity_terminations()
  curriculum = create_velocity_curriculum()

  # Customize actions.
  actions["joint_pos"].scale = GO1_ACTION_SCALE

  # Customize events.
  events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # Customize rewards.
  rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  rewards["foot_swing_height"].params["asset_cfg"].site_names = site_names
  rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  # Disable G1-specific rewards.
  rewards["self_collisions"].weight = 0.0
  rewards["body_ang_vel"].weight = 0.0
  rewards["angular_momentum"].weight = 0.0

  # Customize observations.
  observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  # Customize viewer.
  viewer = deepcopy(VIEWER_CONFIG)
  viewer.body_name = "trunk"
  viewer.distance = 1.5
  viewer.elevation = -10.0

  # Create and return complete config.
  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
    commands=commands,
    curriculum=curriculum,
    sim=SIM_CFG,
    viewer=viewer,
    decimation=4,
    episode_length_s=20.0,
  )


def create_unitree_go1_rough_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain PLAY configuration.

      (infinite episodes, no curriculum).
  """
  cfg = create_unitree_go1_rough_env_cfg()

  # PLAY mode customizations.
  cfg.episode_length_s = int(1e9)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


# Module-level constants for gymnasium registration.
UNITREE_GO1_ROUGH_ENV_CFG = create_unitree_go1_rough_env_cfg()
UNITREE_GO1_ROUGH_ENV_CFG_PLAY = create_unitree_go1_rough_env_cfg_play()
