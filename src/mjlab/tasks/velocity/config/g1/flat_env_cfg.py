"""Unitree G1 flat terrain velocity tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the G1 robot on flat terrain.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.g1.rough_env_cfg import create_unitree_g1_rough_env_cfg


def create_unitree_g1_flat_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity tracking configuration."""
  # Start with rough terrain config.
  cfg = create_unitree_g1_rough_env_cfg()

  # Change to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  return cfg


def create_unitree_g1_flat_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain PLAY configuration."""
  cfg = create_unitree_g1_flat_env_cfg()

  # PLAY mode customizations.
  cfg.episode_length_s = int(1e9)
  cfg.observations["policy"].enable_corruption = False

  assert cfg.events is not None
  assert "push_robot" in cfg.events
  del cfg.events["push_robot"]

  # Higher velocity ranges for PLAY mode.
  assert cfg.commands is not None
  assert "twist" in cfg.commands
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
  twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


# Module-level constants for gymnasium registration.
UNITREE_G1_FLAT_ENV_CFG = create_unitree_g1_flat_env_cfg()
UNITREE_G1_FLAT_ENV_CFG_PLAY = create_unitree_g1_flat_env_cfg_play()
