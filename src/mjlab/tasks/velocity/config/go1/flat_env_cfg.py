from mjlab.tasks.velocity.config.go1.rough_env_cfg import (
  create_unitree_go1_rough_env_cfg,
)


def create_unitree_go1_flat_env_cfg():
  """Create configuration for Unitree GO1 robot on flat terrain."""
  cfg = create_unitree_go1_rough_env_cfg()

  assert cfg.scene.terrain is not None, "Scene terrain must be configured"
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove terrain curriculum for flat terrain
  assert cfg.curriculum is not None, (
    "Curriculum must be configured in base velocity config"
  )
  assert "terrain_levels" in cfg.curriculum, (
    "Terrain levels curriculum should be present in rough config"
  )
  del cfg.curriculum["terrain_levels"]

  return cfg


def create_unitree_go1_flat_env_cfg_play():
  """Create play configuration for Unitree GO1 robot on flat terrain."""
  return create_unitree_go1_flat_env_cfg()


# Create config instances
UNITREE_GO1_FLAT_ENV_CFG = create_unitree_go1_flat_env_cfg()
UNITREE_GO1_FLAT_ENV_CFG_PLAY = create_unitree_go1_flat_env_cfg_play()
