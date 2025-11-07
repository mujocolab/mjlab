"""Unitree G1 rough terrain velocity tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the G1 robot on rough terrain.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.envs import ManagerBasedRlEnvCfg
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


def create_unitree_g1_rough_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity tracking configuration."""
  # G1-specific metadata.
  site_names = ["left_foot", "right_foot"]
  geom_names = []
  for i in range(1, 8):
    geom_names.append(f"left_foot{i}_collision")
  for i in range(1, 8):
    geom_names.append(f"right_foot{i}_collision")
  target_foot_height = 0.15

  # Create scene with G1 robot and sensors.
  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": replace(G1_ROBOT_CFG)}

  # Create sensors (G1 uses subtree mode for contact detection)
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  scene.sensors = (feet_ground_cfg, self_collision_cfg)

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
  actions["joint_pos"].scale = G1_ACTION_SCALE

  # Customize events.
  events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # Customize rewards - G1 has more detailed pose specifications.
  rewards["upright"].params["asset_cfg"].body_names = ["torso_link"]

  # Tight control when stationary: maintain stable default pose.
  rewards["pose"].params["std_standing"] = {".*": 0.05}

  # Moderate leg freedom for stepping, loose arms for natural pendulum swing.
  rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }

  # Maximum freedom for dynamic motion.
  rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  rewards["foot_swing_height"].params["asset_cfg"].site_names = site_names
  rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  rewards["foot_swing_height"].params["target_height"] = target_foot_height
  rewards["foot_clearance"].params["target_height"] = target_foot_height
  rewards["body_ang_vel"].params["asset_cfg"].body_names = ["torso_link"]

  # Customize observations.
  observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # Customize terminations - G1 doesn't use illegal_contact.
  assert "illegal_contact" in terminations
  del terminations["illegal_contact"]

  # Customize viewer.
  viewer = deepcopy(VIEWER_CONFIG)
  viewer.body_name = "torso_link"

  # Customize commands.
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  twist_cmd = commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

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


def create_unitree_g1_rough_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain PLAY configuration."""
  cfg = create_unitree_g1_rough_env_cfg()

  # PLAY mode customizations.
  cfg.episode_length_s = int(1e9)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


# Module-level constants for gymnasium registration.
UNITREE_G1_ROUGH_ENV_CFG = create_unitree_g1_rough_env_cfg()
UNITREE_G1_ROUGH_ENV_CFG_PLAY = create_unitree_g1_rough_env_cfg_play()
