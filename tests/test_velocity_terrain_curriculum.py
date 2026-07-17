"""Tests for the velocity terrain-level curriculum."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel


class _FakeTerrain:
  """Minimal terrain stub implementing the real level-update arithmetic."""

  def __init__(self, num_envs: int, size: float = 8.0):
    self.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_types = torch.zeros(num_envs, dtype=torch.long)
    self.max_terrain_level = 10
    # [num_rows, num_cols, 3]. One column, enough rows for promotion.
    self.terrain_origins = torch.zeros(self.max_terrain_level, 1, 3)
    self.env_origins = torch.zeros(num_envs, 3)
    self.cfg = SimpleNamespace(
      terrain_generator=SimpleNamespace(size=(size, size), sub_terrains={})
    )

  def update_env_origins(self, env_ids, move_up, move_down):
    # Mirror TerrainEntity.update_env_origins exactly, including the cap-wrap
    # branch and the env_origins write, so the fake can't pass vacuously.
    self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    self.terrain_levels[env_ids] = torch.where(
      self.terrain_levels[env_ids] >= self.max_terrain_level,
      torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
      torch.clip(self.terrain_levels[env_ids], 0),
    )
    self.env_origins[env_ids] = self.terrain_origins[
      self.terrain_levels[env_ids], self.terrain_types[env_ids]
    ]


def _make_env(terrain: _FakeTerrain, common_step_counter: int, walked: float):
  num_envs = terrain.terrain_levels.shape[0]
  # Robot has "walked" `walked` meters in x from its origin.
  asset = Mock()
  asset.data.root_link_pos_w = torch.tensor([[walked, 0.0, 0.0]] * num_envs)

  env = Mock()
  env.common_step_counter = common_step_counter
  env.max_episode_length_s = 20.0
  env.scene.__getitem__ = Mock(return_value=asset)
  env.scene.terrain = terrain
  env.scene.env_origins = torch.zeros(num_envs, 3)
  # Zero command so move_down stays inactive; isolates move_up behavior.
  env.command_manager.get_command = Mock(return_value=torch.zeros(num_envs, 2))
  return env


def test_first_reset_does_not_promote_levels():
  """On the initial reset the far spawn/origin gap must not bump levels."""
  terrain = _FakeTerrain(num_envs=4, size=8.0)
  # walked >> size/2 (=4.0), which would normally trigger move_up.
  env = _make_env(terrain, common_step_counter=0, walked=100.0)

  terrain_levels_vel(
    env, torch.arange(4), command_name="twist", asset_cfg=SceneEntityCfg("robot")
  )

  assert torch.all(terrain.terrain_levels == 0)


def test_subsequent_reset_promotes_on_long_walk():
  """After stepping, a long walk still promotes the level as before."""
  terrain = _FakeTerrain(num_envs=4, size=8.0)
  env = _make_env(terrain, common_step_counter=1, walked=100.0)

  terrain_levels_vel(
    env, torch.arange(4), command_name="twist", asset_cfg=SceneEntityCfg("robot")
  )

  assert torch.all(terrain.terrain_levels == 1)
