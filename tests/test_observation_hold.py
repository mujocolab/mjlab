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
  return get_test_device()


@pytest.fixture
def mock_env(device):
  env = Mock()
  env.num_envs = 4
  env.device = device
  env.step_dt = 0.02
  return env


def test_hold_terms_in_sync_and_partial_reset(mock_env, device):
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
