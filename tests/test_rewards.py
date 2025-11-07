"""Tests for reward manager functionality."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.reward_manager import RewardManager


class SimpleTestReward:
  """A simple class-based reward for testing that tracks state."""

  def __init__(self, cfg: RewardTermCfg, env):
    self.num_envs = env.num_envs
    self.device = env.device
    self.current_air_time = torch.zeros((self.num_envs, 1), device=self.device)

  def __call__(self, env, **kwargs):
    self.current_air_time += 0.01
    return torch.ones(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | None = None, env=None):
    if env_ids is not None and len(env_ids) > 0:
      self.current_air_time[env_ids] = 0


class StatelessReward:
  """A stateless class-based reward without reset method."""

  def __init__(self, cfg: RewardTermCfg, env):
    pass

  def __call__(self, env, **kwargs):
    return torch.ones(env.num_envs)


@pytest.fixture
def mock_env():
  """Create a mock environment for testing."""
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.step_dt = 0.01
  env.max_episode_length_s = 10.0
  robot = Mock()
  env.scene = {"robot": robot}
  env.command_manager.get_command = Mock(
    return_value=torch.tensor([[1.0, 0.0, 0.0]] * 4)
  )
  return env


@pytest.fixture
def class_reward_config():
  """Config with a class-based reward."""
  return {
    "term": RewardTermCfg(
      func=SimpleTestReward,
      weight=1.0,
      params={},
    )
  }


@pytest.fixture
def function_reward_config():
  """Config with a function-based reward."""
  return {
    "term": RewardTermCfg(
      func=lambda env: torch.ones(env.num_envs),
      weight=1.0,
      params={},
    )
  }


@pytest.fixture
def stateless_reward_config():
  """Config with a stateless class-based reward."""
  return {
    "term": RewardTermCfg(
      func=StatelessReward,
      weight=1.0,
      params={},
    )
  }


def test_class_based_reward_reset(mock_env, class_reward_config):
  """Test that class-based reward terms are tracked and have reset called."""
  manager = RewardManager(class_reward_config, mock_env)
  term = manager._class_term_cfgs[0].func

  for _ in range(10):
    manager.compute(dt=0.01)
  assert (term.current_air_time > 0).all()

  manager.reset(env_ids=torch.tensor([0, 2]))

  # Check that only specified envs were reset.
  assert term.current_air_time[0, 0] == 0
  assert term.current_air_time[1, 0] > 0
  assert term.current_air_time[2, 0] == 0
  assert term.current_air_time[3, 0] > 0


def test_function_based_reward_not_tracked(mock_env, function_reward_config):
  """Test that function-based reward terms are not tracked as class terms."""
  manager = RewardManager(function_reward_config, mock_env)
  assert len(manager._class_term_cfgs) == 0


def test_stateless_class_reward_no_reset(mock_env, stateless_reward_config):
  """Test that stateless class-based rewards without reset don't break reset."""
  manager = RewardManager(stateless_reward_config, mock_env)

  # Stateless rewards without reset method should not be tracked.
  assert len(manager._class_term_cfgs) == 0

  # Reset should work without errors.
  manager.reset(env_ids=torch.tensor([0, 2]))
