from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.go1.rough_env_cfg import create_go1_rough_env_cfg


def create_go1_flat_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create configuration for Unitree Go1 robot velocity tracking on flat terrain."""
  # Start with rough config and modify for flat terrain.
  cfg = create_go1_rough_env_cfg()

  # Change to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain levels curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  # Reduce push velocity range for flat terrain.
  assert cfg.events is not None
  assert "push_robot" in cfg.events
  push_event = cfg.events["push_robot"]
  push_event.params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
  }

  return cfg


def create_go1_flat_env_cfg_play() -> ManagerBasedRlEnvCfg:
  """Create play configuration for Unitree Go1 robot velocity tracking on flat terrain."""
  cfg = create_go1_flat_env_cfg()

  # Effectively infinite episode length.
  cfg.episode_length_s = int(1e9)

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


UNITREE_GO1_FLAT_ENV_CFG = create_go1_flat_env_cfg()
UNITREE_GO1_FLAT_ENV_CFG_PLAY = create_go1_flat_env_cfg_play()
