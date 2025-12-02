"""Tests for action manager functionality."""

from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.managers.action_manager import ActionManager


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


def _make_mock_action_term(action_dim: int):
  """Create a mock action term factory."""

  def factory(cfg, env):
    term = Mock()
    term.action_dim = action_dim
    term.raw_action = torch.zeros(env.num_envs, action_dim, device=env.device)
    term.process_actions = Mock()
    term.apply_actions = Mock()
    term.reset = Mock()
    return term

  return factory


@pytest.fixture
def mock_env(device):
  """Create a mock environment for testing."""
  env = Mock()
  env.num_envs = 4
  env.device = device
  return env


@pytest.fixture
def action_term_cfg():
  """Create a simple action term config."""
  cfg = Mock()
  cfg.class_type = _make_mock_action_term(action_dim=3)
  cfg.asset_name = "robot"
  return cfg


def test_action_history_tracking(mock_env, action_term_cfg, device):
  """Test that action, prev_action, and prev_prev_action track history correctly."""
  manager = ActionManager({"action": action_term_cfg}, mock_env)

  # Initial state: all zeros.
  assert torch.all(manager.action == 0.0)
  assert torch.all(manager.prev_action == 0.0)
  assert torch.all(manager.prev_prev_action == 0.0)

  # Process actions and verify history shifts correctly.
  actions = [
    torch.tensor([[float(i)] * 3] * mock_env.num_envs, device=device)
    for i in range(1, 5)
  ]

  manager.process_action(actions[0])
  assert torch.allclose(manager.action, actions[0])
  assert torch.all(manager.prev_action == 0.0)
  assert torch.all(manager.prev_prev_action == 0.0)

  manager.process_action(actions[1])
  assert torch.allclose(manager.action, actions[1])
  assert torch.allclose(manager.prev_action, actions[0])
  assert torch.all(manager.prev_prev_action == 0.0)

  manager.process_action(actions[2])
  assert torch.allclose(manager.action, actions[2])
  assert torch.allclose(manager.prev_action, actions[1])
  assert torch.allclose(manager.prev_prev_action, actions[0])

  manager.process_action(actions[3])
  assert torch.allclose(manager.action, actions[3])
  assert torch.allclose(manager.prev_action, actions[2])
  assert torch.allclose(manager.prev_prev_action, actions[1])


def test_action_history_reset(mock_env, action_term_cfg, device):
  """Test that reset clears action history for all or specific environments."""
  manager = ActionManager({"action": action_term_cfg}, mock_env)

  # Populate history.
  actions = [
    torch.tensor([[float(i)] * 3] * mock_env.num_envs, device=device)
    for i in range(1, 4)
  ]
  for a in actions:
    manager.process_action(a)

  # Partial reset: only envs 0 and 2.
  manager.reset(env_ids=torch.tensor([0, 2]))

  # Reset envs should be zeros.
  for env_id in [0, 2]:
    assert torch.all(manager.action[env_id] == 0.0)
    assert torch.all(manager.prev_action[env_id] == 0.0)
    assert torch.all(manager.prev_prev_action[env_id] == 0.0)

  # Non-reset envs should retain history.
  for env_id in [1, 3]:
    assert torch.allclose(manager.action[env_id], actions[2][env_id])
    assert torch.allclose(manager.prev_action[env_id], actions[1][env_id])
    assert torch.allclose(manager.prev_prev_action[env_id], actions[0][env_id])

  # Full reset.
  manager.reset()
  assert torch.all(manager.action == 0.0)
  assert torch.all(manager.prev_action == 0.0)
  assert torch.all(manager.prev_prev_action == 0.0)
