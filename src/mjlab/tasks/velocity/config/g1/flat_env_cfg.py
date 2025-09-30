from mjlab.tasks.velocity.config.g1.rough_env_cfg import create_g1_rough_env_cfg


def create_g1_flat_env_cfg():
  """Create configuration for Unitree G1 robot velocity tracking on flat terrain."""
  # Start with rough config and modify for flat terrain
  cfg = create_g1_rough_env_cfg()

  # Change to flat terrain
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain levels curriculum
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  # Reduce push velocity range for flat terrain
  assert cfg.events is not None
  assert "push_robot" in cfg.events
  push_event = cfg.events["push_robot"]
  push_event.params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
  }

  return cfg


def create_g1_flat_env_cfg_play():
  """Create play configuration for Unitree G1 robot velocity tracking on flat terrain."""
  cfg = create_g1_flat_env_cfg()

  # Effectively infinite episode length
  cfg.episode_length_s = int(1e9)

  return cfg


UNITREE_G1_FLAT_ENV_CFG = create_g1_flat_env_cfg()
UNITREE_G1_FLAT_ENV_CFG_PLAY = create_g1_flat_env_cfg_play()
