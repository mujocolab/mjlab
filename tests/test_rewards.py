from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.reward_manager import RewardManager


@pytest.fixture
def mock_env():
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.scene = {}
  env.command_manager = Mock()
  env.action_manager = Mock()
  env.max_episode_length_s = 10.0
  return env


class FakeResetReward:
  def __init__(self, cfg: RewardTermCfg, env):
    del cfg  # Unused.
    self.values = torch.arange(env.num_envs, dtype=torch.float32, device=env.device)
    self.reset_calls: list[torch.Tensor | slice | None] = []

  def __call__(self, env, **kwargs):
    del env, kwargs  # Unused.
    return self.values

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    self.reset_calls.append(env_ids)


def test_class_based_reward_reset(mock_env):
  @dataclass
  class Cfg:
    term: RewardTermCfg = field(
      default_factory=lambda: RewardTermCfg(
        func=FakeResetReward,
        weight=1.0,
        params={},
      )
    )

  manager = RewardManager(Cfg(), mock_env)
  assert len(manager._class_term_cfgs) == 1

  term_instance = manager._class_term_cfgs[0].func
  dt = 0.1
  manager.compute(dt=dt)

  reset_ids = torch.tensor([0, 2])
  extras = manager.reset(env_ids=reset_ids)

  assert len(term_instance.reset_calls) == 1
  recorded_env_ids = term_instance.reset_calls[0]
  assert torch.equal(recorded_env_ids, reset_ids)

  expected_episode_avg = (
    torch.mean(term_instance.values[reset_ids] * dt).item()
    / mock_env.max_episode_length_s
  )
  actual_episode_avg = extras["Episode_Reward/term"].item()
  assert actual_episode_avg == pytest.approx(expected_episode_avg)


def test_function_based_reward_not_tracked(mock_env):
  @dataclass
  class Cfg:
    term: RewardTermCfg = field(
      default_factory=lambda: RewardTermCfg(
        func=lambda env: torch.ones(env.num_envs, device=env.device),
        weight=1.0,
        params={},
      )
    )

  manager = RewardManager(Cfg(), mock_env)
  assert len(manager._class_term_cfgs) == 0


def test_stateless_class_reward_no_reset(mock_env):
  class StatelessReward:
    def __init__(self, cfg: RewardTermCfg, env):
      del cfg, env  # Unused.

    def __call__(self, env, **kwargs):
      del kwargs  # Unused.
      return torch.ones(env.num_envs, device=env.device)

  @dataclass
  class Cfg:
    term: RewardTermCfg = field(
      default_factory=lambda: RewardTermCfg(
        func=StatelessReward,
        weight=1.0,
        params={},
      )
    )

  manager = RewardManager(Cfg(), mock_env)

  assert len(manager._class_term_cfgs) == 0
  manager.reset(env_ids=torch.tensor([0, 2]))  # Should not raise
