"""Tests for observation noise functionality."""

from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationManager,
  ObservationTermCfg,
)
from mjlab.utils.noise.noise_cfg import ConstantNoiseCfg


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


def test_noise_applied_when_corruption_enabled(mock_env, device):
  """Test that noise is applied when enable_corruption=True."""

  def obs_func(env):
    return torch.ones((env.num_envs, 3), device=device)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=obs_func,
          params={},
          noise=ConstantNoiseCfg(bias=0.5, operation="add"),
        ),
      },
      enable_corruption=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["actor"]
  assert isinstance(policy_obs, torch.Tensor)
  # Expect 1.0 + 0.5 = 1.5
  expected = torch.full((4, 3), 1.5, device=device)
  assert torch.allclose(policy_obs, expected)


def test_noise_not_applied_when_corruption_disabled(mock_env, device):
  """Test that noise is NOT applied when enable_corruption=False."""

  def obs_func(env):
    return torch.ones((env.num_envs, 3), device=device)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=obs_func,
          params={},
          noise=ConstantNoiseCfg(bias=0.5, operation="add"),
        ),
      },
      enable_corruption=False,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["actor"]
  assert isinstance(policy_obs, torch.Tensor)
  # Noise should NOT be applied, expect original value of 1.0
  expected = torch.full((4, 3), 1.0, device=device)
  assert torch.allclose(policy_obs, expected)


def test_noise_add_operation(mock_env, device):
  """Test noise with 'add' operation."""

  def obs_func(env):
    return torch.full((env.num_envs, 3), 2.0, device=device)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=obs_func,
          params={},
          noise=ConstantNoiseCfg(bias=0.3, operation="add"),
        ),
      },
      enable_corruption=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["actor"]
  assert isinstance(policy_obs, torch.Tensor)
  # Expect 2.0 + 0.3 = 2.3
  expected = torch.full((4, 3), 2.3, device=device)
  assert torch.allclose(policy_obs, expected)


def test_noise_scale_operation(mock_env, device):
  """Test noise with 'scale' operation."""

  def obs_func(env):
    return torch.full((env.num_envs, 3), 2.0, device=device)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=obs_func,
          params={},
          noise=ConstantNoiseCfg(bias=0.5, operation="scale"),
        ),
      },
      enable_corruption=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["actor"]
  assert isinstance(policy_obs, torch.Tensor)
  # Expect 2.0 * 0.5 = 1.0
  expected = torch.full((4, 3), 1.0, device=device)
  assert torch.allclose(policy_obs, expected)


def test_noise_abs_operation(mock_env, device):
  """Test noise with 'abs' operation (replaces data with bias)."""

  def obs_func(env):
    return torch.full((env.num_envs, 3), 2.0, device=device)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=obs_func,
          params={},
          noise=ConstantNoiseCfg(bias=0.7, operation="abs"),
        ),
      },
      enable_corruption=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["actor"]
  assert isinstance(policy_obs, torch.Tensor)
  # Expect data to be replaced with bias = 0.7
  expected = torch.full((4, 3), 0.7, device=device)
  assert torch.allclose(policy_obs, expected)


def test_noise_with_per_dimension_bias(mock_env, device):
  """Test noise with per-dimension bias (tuple)."""

  def obs_func(env):
    return torch.ones((env.num_envs, 3), device=device)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "obs1": ObservationTermCfg(
          func=obs_func,
          params={},
          noise=ConstantNoiseCfg(bias=(0.1, 0.2, 0.3), operation="add"),
        ),
      },
      enable_corruption=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["actor"]
  assert isinstance(policy_obs, torch.Tensor)
  # Expect 1.0 + [0.1, 0.2, 0.3] = [1.1, 1.2, 1.3]
  expected = torch.tensor([[1.1, 1.2, 1.3]] * 4, device=device)
  assert torch.allclose(policy_obs, expected)


def test_noise_tensor_caching(device):
  """Test that tensor conversion is cached across multiple calls."""
  noise = ConstantNoiseCfg(bias=0.5)
  data = torch.ones((4, 3), device=device)

  # First call should create the tensor
  result1 = noise.apply(data)

  # Verify the cache was populated and get reference to cached tensor
  device_key = str(result1.device)
  assert device_key in noise._tensor_cache
  assert "bias" in noise._tensor_cache[device_key]
  cached_tensor = noise._tensor_cache[device_key]["bias"]
  cached_data_ptr = cached_tensor.data_ptr()

  # Second call should use the same cached tensor
  result2 = noise.apply(data)

  # Verify the cached tensor is the same object (same memory address)
  assert noise._tensor_cache[device_key]["bias"].data_ptr() == cached_data_ptr

  # Both results should be correct
  expected = torch.full((4, 3), 1.5, device=device)
  assert torch.allclose(result1, expected)
  assert torch.allclose(result2, expected)


def test_multiple_terms_with_different_noise(mock_env, device):
  """Test multiple observation terms with different noise configs."""

  def obs_func_a(env):
    return torch.ones((env.num_envs, 2), device=device)

  def obs_func_b(env):
    return torch.full((env.num_envs, 2), 3.0, device=device)

  cfg = {
    "actor": ObservationGroupCfg(
      terms={
        "obs_a": ObservationTermCfg(
          func=obs_func_a,
          params={},
          noise=ConstantNoiseCfg(bias=0.1, operation="add"),
        ),
        "obs_b": ObservationTermCfg(
          func=obs_func_b,
          params={},
          noise=ConstantNoiseCfg(bias=2.0, operation="scale"),
        ),
      },
      enable_corruption=True,
    ),
  }

  manager = ObservationManager(cfg, mock_env)
  obs = manager.compute()

  policy_obs = obs["actor"]
  assert isinstance(policy_obs, torch.Tensor)
  # obs_a: 1.0 + 0.1 = 1.1
  # obs_b: 3.0 * 2.0 = 6.0
  # Concatenated: [1.1, 1.1, 6.0, 6.0]
  expected = torch.tensor([[1.1, 1.1, 6.0, 6.0]] * 4, device=device)
  assert torch.allclose(policy_obs, expected)
