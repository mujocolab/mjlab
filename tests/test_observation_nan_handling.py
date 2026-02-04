"""Tests for observation NaN handling functionality."""

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
  """Create a mock environment."""
  env = Mock()
  env.num_envs = 4
  env.device = device
  env.step_dt = 0.02
  return env


def test_nan_disabled_policy_no_check(mock_env, device):
  """Policy='disabled' should pass through NaNs unchanged."""

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    obs[1, 1] = float("nan")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=obs_with_nan, params={})},
      nan_policy="disabled",
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert torch.isnan(policy_obs[1, 1])  # NaN should pass through.


def test_nan_sanitize_policy_silently_fixes(mock_env, device):
  """Policy='sanitize' should zero NaNs without warning."""

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    obs[1, 1] = float("nan")
    obs[2, 0] = float("inf")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=obs_with_nan, params={})},
      nan_policy="sanitize",
      nan_check_per_term=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert not torch.isnan(policy_obs).any()
  assert not torch.isinf(policy_obs).any()
  assert policy_obs[1, 1] == 0.0
  assert policy_obs[2, 0] == 0.0


def test_nan_warn_policy_logs_and_sanitizes(mock_env, device, capsys):
  """Policy='warn' should log warning and sanitize."""

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    obs[1, 1] = float("nan")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=obs_with_nan, params={})},
      nan_policy="warn",
      nan_check_per_term=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  captured = capsys.readouterr()
  assert "ObservationManager" in captured.out
  assert "NaN/Inf" in captured.out
  assert "policy/obs1" in captured.out
  assert "Sanitizing" in captured.out

  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert not torch.isnan(policy_obs).any()
  assert policy_obs[1, 1] == 0.0


def test_nan_error_policy_raises(mock_env, device):
  """Policy='error' should raise ValueError with context."""

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    obs[1, 1] = float("nan")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=obs_with_nan, params={})},
      nan_policy="error",
      nan_check_per_term=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  with pytest.raises(ValueError) as excinfo:
    manager.compute()

  assert "NaN/Inf detected" in str(excinfo.value)
  assert "policy/obs1" in str(excinfo.value)
  assert "1" in str(excinfo.value)  # Environment ID.


def test_nan_check_per_term_identifies_source(mock_env, device, capsys):
  """Per-term checking should identify which term has NaN."""

  def clean_obs(env):
    return torch.ones((env.num_envs, 3), device=device)

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    obs[1, 1] = float("nan")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "clean": ObservationTermCfg(func=clean_obs, params={}),
        "problematic": ObservationTermCfg(func=obs_with_nan, params={}),
      },
      nan_policy="warn",
      nan_check_per_term=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  # Check that only problematic term is logged.
  captured = capsys.readouterr()
  assert "policy/problematic" in captured.out
  assert "policy/clean" not in captured.out

  # Both terms should be sanitized in output.
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert not torch.isnan(policy_obs).any()


def test_nan_check_final_only(mock_env, device, capsys):
  """Final-only checking should check concatenated result."""

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    obs[1, 1] = float("nan")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=obs_with_nan, params={})},
      nan_policy="warn",
      nan_check_per_term=False,  # Check only final result.
      concatenate_terms=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  # Should log group name, not term name.
  captured = capsys.readouterr()
  assert "policy" in captured.out
  assert "policy/obs1" not in captured.out  # Should not show term.

  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert not torch.isnan(policy_obs).any()


def test_nan_check_dict_output(mock_env, device):
  """NaN checking should work with non-concatenated dict output."""

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    obs[1, 1] = float("nan")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=obs_with_nan, params={})},
      nan_policy="sanitize",
      concatenate_terms=False,
      nan_check_per_term=False,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy: dict[str, torch.Tensor] = obs["policy"]  # type: ignore[assignment]
  policy_obs = policy["obs1"]
  assert not torch.isnan(policy_obs).any()
  assert policy_obs[1, 1] == 0.0


def test_nan_before_delay_buffer(mock_env, device):
  """NaN should be sanitized before entering delay buffer."""

  call_count = [0]

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    # Only inject NaN on first call.
    if call_count[0] == 0:
      obs[1, 1] = float("nan")
    call_count[0] += 1
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=obs_with_nan,
          params={},
          delay_min_lag=1,
          delay_max_lag=1,  # 1-step delay.
        )
      },
      nan_policy="sanitize",
      nan_check_per_term=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  # First compute: NaN should be sanitized before entering delay buffer.
  obs1 = manager.compute(update_history=True)
  # First output comes from buffer initialization (zeros).
  obs1_policy = obs1["policy"]
  assert isinstance(obs1_policy, torch.Tensor)
  assert not torch.isnan(obs1_policy).any()

  # Second compute: should get sanitized value from buffer.
  obs2 = manager.compute(update_history=True)
  obs2_policy = obs2["policy"]
  assert isinstance(obs2_policy, torch.Tensor)
  assert not torch.isnan(obs2_policy).any()


def test_nan_before_history_buffer(mock_env, device):
  """NaN should be sanitized before entering history buffer."""

  call_count = [0]

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    # Inject NaN on first call.
    if call_count[0] == 0:
      obs[1, 1] = float("nan")
    call_count[0] += 1
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=obs_with_nan, params={}, history_length=3, flatten_history_dim=True
        )
      },
      nan_policy="sanitize",
      nan_check_per_term=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)

  # First compute: NaN should be sanitized.
  obs1 = manager.compute(update_history=True)
  obs1_policy = obs1["policy"]
  assert isinstance(obs1_policy, torch.Tensor)
  assert not torch.isnan(obs1_policy).any()

  # Subsequent computes should also have no NaN from history.
  for _ in range(3):
    obs = manager.compute(update_history=True)
    obs_policy = obs["policy"]
    assert isinstance(obs_policy, torch.Tensor)
    assert not torch.isnan(obs_policy).any()


def test_nan_handling_with_multiple_groups(mock_env, device):
  """NaN handling should work independently for multiple groups."""

  def obs_with_nan(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    obs[1, 1] = float("nan")
    return obs

  def clean_obs(env):
    return torch.ones((env.num_envs, 3), device=device)

  cfg = {
    "policy": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=obs_with_nan, params={})},
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=clean_obs, params={})},
      nan_policy="disabled",
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert not torch.isnan(policy_obs).any()

  critic_obs = obs["critic"]
  assert isinstance(critic_obs, torch.Tensor)
  assert not torch.isnan(critic_obs).any()


def test_nan_handling_with_scaling(mock_env, device):
  """NaN handling should work correctly with scaling applied."""

  def obs_with_nan(env):
    obs = torch.full((env.num_envs, 3), 2.0, device=device)
    obs[1, 1] = float("nan")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=obs_with_nan, params={}, scale=0.5)},
      nan_policy="sanitize",
      nan_check_per_term=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  # NaN should be sanitized to 0, other values scaled: 2.0 * 0.5 = 1.0.
  assert not torch.isnan(policy_obs).any()
  assert policy_obs[1, 1] == 0.0
  assert policy_obs[0, 0] == 1.0  # Scaled value.


def test_nan_handling_with_clipping(mock_env, device):
  """NaN handling should work correctly with clipping applied."""

  def obs_with_nan(env):
    obs = torch.full((env.num_envs, 3), 10.0, device=device)
    obs[1, 1] = float("nan")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(func=obs_with_nan, params={}, clip=(-1.0, 1.0))
      },
      nan_policy="sanitize",
      nan_check_per_term=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  # NaN should be sanitized to 0, other values clipped to 1.0.
  assert not torch.isnan(policy_obs).any()
  assert policy_obs[1, 1] == 0.0
  assert policy_obs[0, 0] == 1.0  # Clipped value.


def test_negative_inf_handling(mock_env, device):
  """Test that negative infinity is also handled."""

  def obs_with_neginf(env):
    obs = torch.ones((env.num_envs, 3), device=device)
    obs[1, 1] = float("-inf")
    return obs

  cfg = {
    "policy": ObservationGroupCfg(
      terms={"obs1": ObservationTermCfg(func=obs_with_neginf, params={})},
      nan_policy="sanitize",
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert not torch.isinf(policy_obs).any()
  assert policy_obs[1, 1] == 0.0
