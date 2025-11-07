"""Unitree Go1 flat terrain velocity tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the Go1 robot on flat terrain.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.go1.rough_env_cfg import (
  create_unitree_go1_rough_env_cfg,
)


def create_unitree_go1_flat_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 flat terrain velocity tracking configuration."""
  # Start with rough terrain config.
  cfg = create_unitree_go1_rough_env_cfg()

  # Change to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  return cfg


def create_unitree_go1_flat_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 flat terrain PLAY configuration."""
  cfg = create_unitree_go1_flat_env_cfg()

  # PLAY mode customizations.
  cfg.episode_length_s = int(1e9)
  cfg.observations["policy"].enable_corruption = False
  cfg.events["push_robot"] = None

  return cfg


# Module-level constants for gymnasium registration.
UNITREE_GO1_FLAT_ENV_CFG = create_unitree_go1_flat_env_cfg()
UNITREE_GO1_FLAT_ENV_CFG_PLAY = create_unitree_go1_flat_env_cfg_play()
