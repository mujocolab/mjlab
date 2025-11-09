"""Unitree Go1 rough terrain velocity tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the Go1 robot on rough terrain.
"""

from copy import deepcopy

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import VIEWER_CONFIG, create_velocity_env_cfg


def create_unitree_go1_rough_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain velocity tracking configuration."""
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

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

  cfg = create_velocity_env_cfg(
    robot_cfg=get_go1_robot_cfg(),
    action_scale=GO1_ACTION_SCALE,
    viewer_body_name="trunk",
    site_names=site_names,
    feet_sensor_cfg=feet_ground_cfg,
    self_collision_sensor_cfg=nonfoot_ground_cfg,
    foot_friction_geom_names=geom_names,
    posture_std_standing={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
    },
    posture_std_walking={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    },
    posture_std_running={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    },
  )

  cfg.viewer = deepcopy(VIEWER_CONFIG)
  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  assert cfg.terminations is not None
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "nonfoot_ground_touch"},
  )

  return cfg


def create_unitree_go1_rough_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain PLAY configuration (infinite episodes, no curriculum)."""
  cfg = create_unitree_go1_rough_env_cfg()

  cfg.episode_length_s = int(1e9)

  assert (
    cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
  )
  cfg.scene.terrain.terrain_generator.curriculum = False
  cfg.scene.terrain.terrain_generator.num_cols = 5
  cfg.scene.terrain.terrain_generator.num_rows = 5
  cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


UNITREE_GO1_ROUGH_ENV_CFG = create_unitree_go1_rough_env_cfg()
UNITREE_GO1_ROUGH_ENV_CFG_PLAY = create_unitree_go1_rough_env_cfg_play()
