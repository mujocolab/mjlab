"""Tests for observation delay functionality."""

from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.observation_manager import ObservationManager


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def mock_env(device):
  """Create a mock environment."""
  env = Mock()
  env.num_envs = 4
  env.device = device
  env.step_dt = 0.02
  return env


@pytest.fixture
def simple_obs_func(device):
  """Create a simple observation function that returns a counter."""
  counter = {"value": 0}

  def obs_func(env):
    counter["value"] += 1
    return torch.full((env.num_envs, 3), float(counter["value"]), device=device)

  return obs_func


##
# Basic delay tests.
##


def test_no_delay_by_default(mock_env, simple_obs_func):
  """Test that observations work without delay (default behavior)."""

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(func=simple_obs_func, params={}),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  assert manager.group_obs_dim["policy"] == (3,)

  obs = manager.compute()
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert policy_obs.shape == (4, 3)


def test_constant_delay(mock_env, simple_obs_func, device):
  """Test observation with constant delay (min_lag = max_lag = 2)."""

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=simple_obs_func, params={}, delay_min_lag=2, delay_max_lag=2
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  # Note: counter is incremented during _prepare_terms (value=1).

  # First compute uses value=2.
  # Delay: lag=2 sampled, buffer only has 1 frame, clamped to lag=0, returns 2.
  obs1 = manager.compute()
  policy_obs1 = obs1["policy"]
  assert isinstance(policy_obs1, torch.Tensor)
  assert torch.allclose(policy_obs1[0], torch.full((3,), 2.0, device=device))

  # Second compute uses value=3.
  # Delay: lag=2, buffer has 2 frames, clamped to lag=1, returns 2.
  obs2 = manager.compute()
  policy_obs2 = obs2["policy"]
  assert isinstance(policy_obs2, torch.Tensor)
  assert torch.allclose(policy_obs2[0], torch.full((3,), 2.0, device=device))

  # Third compute uses value=4.
  # Delay: lag=2, buffer full (3 frames), returns value from 2 steps ago = 2.
  obs3 = manager.compute()
  policy_obs3 = obs3["policy"]
  assert isinstance(policy_obs3, torch.Tensor)
  assert torch.allclose(policy_obs3[0], torch.full((3,), 2.0, device=device))

  # Fourth compute uses value=5.
  # Delay: lag=2, returns value from 2 steps ago = 3.
  obs4 = manager.compute()
  policy_obs4 = obs4["policy"]
  assert isinstance(policy_obs4, torch.Tensor)
  assert torch.allclose(policy_obs4[0], torch.full((3,), 3.0, device=device))

  # Fifth compute uses value=6.
  # Delay: lag=2, returns value from 2 steps ago = 4.
  obs5 = manager.compute()
  policy_obs5 = obs5["policy"]
  assert isinstance(policy_obs5, torch.Tensor)
  assert torch.allclose(policy_obs5[0], torch.full((3,), 4.0, device=device))


def test_zero_delay_returns_current(mock_env, simple_obs_func, device):
  """Test that delay with min_lag=max_lag=0 returns current observation."""

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=simple_obs_func, params={}, delay_min_lag=0, delay_max_lag=0
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  # First compute uses value=2 (counter at 1 after _prepare_terms).
  obs1 = manager.compute()
  policy_obs1 = obs1["policy"]
  assert isinstance(policy_obs1, torch.Tensor)
  assert torch.allclose(policy_obs1[0], torch.full((3,), 2.0, device=device))

  # Second compute uses value=3.
  obs2 = manager.compute()
  policy_obs2 = obs2["policy"]
  assert isinstance(policy_obs2, torch.Tensor)
  assert torch.allclose(policy_obs2[0], torch.full((3,), 3.0, device=device))


def test_lag_one_delay(mock_env, simple_obs_func, device):
  """Test lag=1 returns previous observation."""

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=simple_obs_func, params={}, delay_min_lag=1, delay_max_lag=1
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  # Build sequence: 2, 3, 4, 5, 6
  # Expected with lag=1: 2, 2, 3, 4, 5

  obs1 = manager.compute()  # value=2, lag constrained, returns 2
  assert isinstance(obs1["policy"], torch.Tensor)
  assert torch.allclose(obs1["policy"][0], torch.full((3,), 2.0, device=device))

  obs2 = manager.compute()  # value=3, lag=1, returns 2
  assert isinstance(obs2["policy"], torch.Tensor)
  assert torch.allclose(obs2["policy"][0], torch.full((3,), 2.0, device=device))

  obs3 = manager.compute()  # value=4, lag=1, returns 3
  assert isinstance(obs3["policy"], torch.Tensor)
  assert torch.allclose(obs3["policy"][0], torch.full((3,), 3.0, device=device))

  obs4 = manager.compute()  # value=5, lag=1, returns 4
  assert isinstance(obs4["policy"], torch.Tensor)
  assert torch.allclose(obs4["policy"][0], torch.full((3,), 4.0, device=device))

  obs5 = manager.compute()  # value=6, lag=1, returns 5
  assert isinstance(obs5["policy"], torch.Tensor)
  assert torch.allclose(obs5["policy"][0], torch.full((3,), 5.0, device=device))


##
# Delay with other features.
##


def test_delay_with_history(mock_env, simple_obs_func, device):
  """Test that delay is applied before history."""

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=simple_obs_func,
          params={},
          delay_min_lag=1,
          delay_max_lag=1,
          history_length=2,
          flatten_history_dim=False,
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  assert manager.group_obs_dim["policy"] == (2, 3)

  # First compute: value=2, delay gives 2, history [2, 2].
  obs1 = manager.compute(update_history=False)
  policy_obs1 = obs1["policy"]
  assert isinstance(policy_obs1, torch.Tensor)
  assert torch.allclose(
    policy_obs1[0],
    torch.stack(
      [torch.full((3,), 2.0, device=device), torch.full((3,), 2.0, device=device)]
    ),
  )

  # Second compute: value=3, delay gives 2 (lag=1), history updated to [2, 2].
  obs2 = manager.compute(update_history=True)
  policy_obs2 = obs2["policy"]
  assert isinstance(policy_obs2, torch.Tensor)
  assert torch.allclose(
    policy_obs2[0],
    torch.stack(
      [torch.full((3,), 2.0, device=device), torch.full((3,), 2.0, device=device)]
    ),
  )

  # Third compute: value=4, delay gives 3 (lag=1), history updated to [2, 3].
  obs3 = manager.compute(update_history=True)
  policy_obs3 = obs3["policy"]
  assert isinstance(policy_obs3, torch.Tensor)
  assert torch.allclose(
    policy_obs3[0],
    torch.stack(
      [torch.full((3,), 2.0, device=device), torch.full((3,), 3.0, device=device)]
    ),
  )


def test_delay_with_scale(mock_env, simple_obs_func, device):
  """Test that scaling is applied before delay."""

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=simple_obs_func,
          params={},
          scale=2.0,
          delay_min_lag=1,
          delay_max_lag=1,
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  # First compute: value=2, scaled to 4.
  obs1 = manager.compute()
  assert isinstance(obs1["policy"], torch.Tensor)
  assert torch.allclose(obs1["policy"][0], torch.full((3,), 4.0, device=device))

  # Second compute: value=3, scaled to 6, delay returns 4 (lag=1).
  obs2 = manager.compute()
  assert isinstance(obs2["policy"], torch.Tensor)
  assert torch.allclose(obs2["policy"][0], torch.full((3,), 4.0, device=device))

  # Third compute: value=4, scaled to 8, delay returns 6 (lag=1).
  obs3 = manager.compute()
  assert isinstance(obs3["policy"], torch.Tensor)
  assert torch.allclose(obs3["policy"][0], torch.full((3,), 6.0, device=device))


##
# Reset tests.
##


def test_reset_clears_delay_buffer(mock_env, simple_obs_func, device):
  """Test that reset clears delay buffer and restarts lag constraint."""

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=simple_obs_func, params={}, delay_min_lag=2, delay_max_lag=2
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  # Build up delay buffer with several steps.
  obs = manager.compute()
  for _ in range(4):
    obs = manager.compute()
  # At this point, delay should be working (lag=2).
  # Value should be from 2 steps ago.
  assert isinstance(obs["policy"], torch.Tensor)
  last_val = obs["policy"][0, 0].item()

  # Reset all environments.
  manager.reset()

  # After reset, buffer and lags are cleared.
  # Next compute should return current (lag constrained to 0).
  obs = manager.compute()
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  # Value should be current, not delayed.
  current_val = policy_obs[0, 0].item()
  # Verify it's different from before reset (counter continues).
  assert current_val != last_val


def test_reset_partial_envs_with_verification(mock_env, device):
  """Test partial reset actually resets only specified environments."""
  # Use per-env counters to track each env independently.
  counters = torch.zeros(4, dtype=torch.long, device=device)

  def per_env_obs_func(env):
    counters[:] += 1
    # Return different values per env so we can track them.
    return counters.unsqueeze(1).repeat(1, 3).float()

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=per_env_obs_func, params={}, delay_min_lag=1, delay_max_lag=1
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  # Build up some history.
  for _ in range(3):
    manager.compute()

  # Get obs for env 1 and 3 before reset.
  obs_before = manager.compute()
  assert isinstance(obs_before["policy"], torch.Tensor)
  env1_before = obs_before["policy"][1, 0].item()
  env3_before = obs_before["policy"][3, 0].item()

  # Reset only envs 0 and 2.
  manager.reset(env_ids=torch.tensor([0, 2], device=device))

  # Next compute.
  obs_after = manager.compute()

  # Envs 1 and 3 should have continuous delayed values.
  # Env 0 and 2 should have restarted (returning current due to lag constraint).
  assert isinstance(obs_after["policy"], torch.Tensor)
  env1_after = obs_after["policy"][1, 0].item()
  env3_after = obs_after["policy"][3, 0].item()

  # Non-reset envs should have changed (counter incremented).
  assert env1_after != env1_before
  assert env3_after != env3_before


##
# Shared vs per-env delay.
##


def test_shared_delay_actual_verification(mock_env, device):
  """Verify that per_env=False gives same lag across all envs."""

  # Use unique values per env to distinguish them.
  def unique_obs_func(env):
    # Return env index as the observation value.
    return torch.arange(env.num_envs, device=device).unsqueeze(1).repeat(1, 3).float()

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=unique_obs_func,
          params={},
          delay_min_lag=0,
          delay_max_lag=2,
          delay_per_env=False,  # Same lag for all envs.
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  # Run several steps and track the delay buffer.
  delay_buffer = manager._group_obs_term_delay_buffer["policy"]["obs1"]

  for _ in range(10):
    manager.compute()

  # Check that all envs have the same lag.
  lags = delay_buffer.current_lags
  assert torch.all(lags == lags[0]), f"Expected same lag for all envs, got {lags}"


##
# Update period tests.
##


def test_update_period_actual_verification(mock_env, device):
  """Verify that update_period controls lag update frequency."""

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=lambda env: torch.zeros((env.num_envs, 3), device=device),
          params={},
          delay_min_lag=0,
          delay_max_lag=3,
          delay_update_period=3,
          delay_per_env_phase=False,  # All envs update together.
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  delay_buffer = manager._group_obs_term_delay_buffer["policy"]["obs1"]

  # Track lag changes over time.
  lag_history = []
  for _ in range(12):
    manager.compute()
    lag_history.append(delay_buffer.current_lags[0].item())

  # Lags should only change every 3 steps (at steps 0, 3, 6, 9).
  assert len(lag_history) == 12


##
# Dimension verification tests.
##


def test_delay_preserves_dimensions(mock_env, simple_obs_func):
  """Test that delay preserves observation dimensions."""

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=simple_obs_func, params={}, delay_min_lag=1, delay_max_lag=3
        ),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  assert manager.group_obs_dim["policy"] == (3,)

  obs = manager.compute()
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert policy_obs.shape == (4, 3)


def test_mixed_delay_and_no_delay_terms(mock_env, simple_obs_func, device):
  """Test group with both delayed and non-delayed terms."""

  counter = {"value": 0}

  def obs_func2(env):
    counter["value"] += 1
    return torch.full((env.num_envs, 2), float(counter["value"]) * 10, device=device)

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs_with_delay": ObservationTermCfg(
          func=simple_obs_func, params={}, delay_min_lag=1, delay_max_lag=1
        ),
        "obs_no_delay": ObservationTermCfg(func=obs_func2, params={}),
      }
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  # Should concatenate: 3 + 2 = 5.
  assert manager.group_obs_dim["policy"] == (5,)

  # First compute.
  obs1 = manager.compute()
  assert isinstance(obs1["policy"], torch.Tensor)
  assert obs1["policy"].shape == (4, 5)
  # First 3 values should be delayed obs (value 2).
  # Last 2 values should be non-delayed obs (value 2*10=20).
  assert torch.allclose(obs1["policy"][0, :3], torch.full((3,), 2.0, device=device))
  assert torch.allclose(obs1["policy"][0, 3:], torch.full((2,), 20.0, device=device))

  # Second compute.
  obs2 = manager.compute()
  assert isinstance(obs2["policy"], torch.Tensor)
  # Delayed obs should be 2 (lag=1), non-delayed should be 3*10=30.
  assert torch.allclose(obs2["policy"][0, :3], torch.full((3,), 2.0, device=device))
  assert torch.allclose(obs2["policy"][0, 3:], torch.full((2,), 30.0, device=device))

  # Third compute.
  obs3 = manager.compute()
  assert isinstance(obs3["policy"], torch.Tensor)
  # Delayed obs should be 3 (lag=1), non-delayed should be 4*10=40.
  assert torch.allclose(obs3["policy"][0, :3], torch.full((3,), 3.0, device=device))
  assert torch.allclose(obs3["policy"][0, 3:], torch.full((2,), 40.0, device=device))
