"""Unitree G1 rough terrain velocity tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the G1 robot on rough terrain.
"""

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg


def create_unitree_g1_rough_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity tracking configuration."""
  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

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

  cfg = create_velocity_env_cfg(
    robot_cfg=get_g1_robot_cfg(),
    action_scale=G1_ACTION_SCALE,
    viewer_body_name="torso_link",
    site_names=site_names,
    feet_sensor_cfg=feet_ground_cfg,
    self_collision_sensor_cfg=self_collision_cfg,
    foot_friction_geom_names=geom_names,
    posture_std_standing={".*": 0.05},
    posture_std_walking={
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      r".*waist_yaw.*": 0.2,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.1,
      r".*shoulder_pitch.*": 0.15,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.15,
      r".*wrist.*": 0.3,
    },
    posture_std_running={
      r".*hip_pitch.*": 0.5,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.6,
      r".*ankle_pitch.*": 0.35,
      r".*ankle_roll.*": 0.15,
      r".*waist_yaw.*": 0.3,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.2,
      r".*shoulder_pitch.*": 0.5,
      r".*shoulder_roll.*": 0.2,
      r".*shoulder_yaw.*": 0.15,
      r".*elbow.*": 0.35,
      r".*wrist.*": 0.3,
    },
    body_ang_vel_weight=-0.05,
    angular_momentum_weight=-0.02,
    self_collision_weight=-1.0,
  )

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  if isinstance(twist_cmd, UniformVelocityCommandCfg):
    twist_cmd.viz.z_offset = 1.15

  return cfg


def create_unitree_g1_rough_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain PLAY configuration."""
  cfg = create_unitree_g1_rough_env_cfg()

  cfg.episode_length_s = int(1e9)

  assert (
    cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
  )
  cfg.scene.terrain.terrain_generator.curriculum = False
  cfg.scene.terrain.terrain_generator.num_cols = 5
  cfg.scene.terrain.terrain_generator.num_rows = 5
  cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


UNITREE_G1_ROUGH_ENV_CFG = create_unitree_g1_rough_env_cfg()
UNITREE_G1_ROUGH_ENV_CFG_PLAY = create_unitree_g1_rough_env_cfg_play()
