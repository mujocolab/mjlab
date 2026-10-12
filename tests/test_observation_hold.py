"""Tests for observation sample-and-hold (sample_rate_hz)."""

from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationManager,
  ObservationTermCfg,
)


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def mock_env(device):
  """Create a mock environment with a 50 Hz control loop."""
  env = Mock()
  env.num_envs = 4
  env.device = device
  env.step_dt = 0.02
  return env


def test_hold_terms_in_sync_and_partial_reset(mock_env, device):
  """Terms with the same rate refresh together, and a partial reset refreshes only
  the reset env."""
  value = {"v": 0.0}

  def obs_func(env):
    return torch.full((env.num_envs, 1), value["v"], device=device)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "a": ObservationTermCfg(func=obs_func, params={}, sample_rate_hz=25.0),
        "b": ObservationTermCfg(func=obs_func, params={}, sample_rate_hz=25.0),
        "fast": ObservationTermCfg(func=obs_func, params={}),
      }
    ),
  }
  manager = ObservationManager(cfg, mock_env)
  held = torch.empty(0)
  for step in range(6):
    value["v"] = float(step)
    obs = manager.compute(update_history=True)["actor"]
    assert isinstance(obs, torch.Tensor)
    assert torch.equal(obs[:, 0], obs[:, 1])
    assert (obs[:, 2] == step).all()
    held = obs[:, 0].clone()
  # 25 Hz on a 50 Hz loop: every env refreshed on step 4 or 5.
  assert ((held == 4) | (held == 5)).all()

  env_ids = torch.tensor([1], device=device)
  manager.reset(env_ids=env_ids)
  value["v"] = 10.0
  obs = manager.compute(update_history=True, env_ids=env_ids)["actor"]
  assert isinstance(obs, torch.Tensor)
  assert obs[1, 0] == 10.0
  others = torch.tensor([0, 2, 3], device=device)
  assert torch.equal(obs[others, 0], held[others])


def test_hold_feeds_delay_and_history(mock_env, device):
  """Delay and history operate on the held stream."""
  value = {"v": 0.0}

  def obs_func(env):
    return torch.full((env.num_envs, 1), value["v"], device=device)

  def term(**kwargs) -> ObservationTermCfg:
    return ObservationTermCfg(func=obs_func, params={}, sample_rate_hz=30.0, **kwargs)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "held": term(),
        "delayed": term(delay_min_lag=1, delay_max_lag=1),
        "hist": term(history_length=3),
      }
    ),
  }
  manager = ObservationManager(cfg, mock_env)
  held: list[torch.Tensor] = []
  for step in range(10):
    value["v"] = float(step)
    obs = manager.compute(update_history=True)["actor"]
    assert isinstance(obs, torch.Tensor)
    held.append(obs[:, 0])
    assert torch.equal(obs[:, 1], held[max(step - 1, 0)])
    expected = torch.stack([held[max(step - i, 0)] for i in (2, 1, 0)], dim=1)
    assert torch.equal(obs[:, 2:5], expected)


def test_hold_rejects_non_positive_rate(mock_env, device):
  """A non-positive rate fails at construction and names the term."""

  def obs_func(env):
    return torch.zeros((env.num_envs, 1), device=device)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={"cam": ObservationTermCfg(func=obs_func, params={}, sample_rate_hz=0.0)}
    ),
  }
  with pytest.raises(ValueError, match="'cam' in group 'actor'"):
    ObservationManager(cfg, mock_env)
