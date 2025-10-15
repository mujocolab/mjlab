"""Tests for observation history functionality."""

from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.observation_manager import ObservationManager


def get_test_device():
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda:0"
  return "cpu"


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


# Basic observation history tests.


def test_no_history_by_default(mock_env, simple_obs_func):
  """Test that observations work without history (default behavior)."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(func=simple_obs_func, params={})
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)
  assert manager.group_obs_dim["policy"] == (3,)

  obs = manager.compute()
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert policy_obs.shape == (4, 3)


def test_single_step_history(mock_env, simple_obs_func):
  """Test observation with history_length=1."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func, params={}, history_length=1
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)
  assert manager.group_obs_dim["policy"] == (3,)

  obs = manager.compute()
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert policy_obs.shape == (4, 3)


def test_multi_step_history_flattened(mock_env, simple_obs_func):
  """Test observation with history_length=3 and flattened."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func, params={}, history_length=3, flatten_history_dim=True
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)
  assert manager.group_obs_dim["policy"] == (9,)

  obs = manager.compute()
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert policy_obs.shape == (4, 9)


def test_multi_step_history_not_flattened(mock_env, simple_obs_func):
  """Test observation with history_length=3 and not flattened."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func, params={}, history_length=3, flatten_history_dim=False
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)
  assert manager.group_obs_dim["policy"] == (3, 3)

  obs = manager.compute()
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert policy_obs.shape == (4, 3, 3)


# History accumulation tests.


def test_history_accumulates_correctly(mock_env, simple_obs_func):
  """Test that history buffer accumulates observations in correct order."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func, params={}, history_length=3, flatten_history_dim=False
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)
  device = mock_env.device
  # Note: counter is incremented during _prepare_terms (value=1).

  # First compute uses value=2 and initializes buffer.
  obs = manager.compute(update_history=False)
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert torch.allclose(policy_obs[0], torch.full((3, 3), 2.0, device=device))

  # Update with value=3.
  obs = manager.compute(update_history=True)
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  # History: [2, 2, 3] (oldest to newest).
  expected = torch.stack(
    [
      torch.full((3,), 2.0, device=device),
      torch.full((3,), 2.0, device=device),
      torch.full((3,), 3.0, device=device),
    ]
  )
  assert torch.allclose(policy_obs[0], expected)

  # Update with value=4.
  obs = manager.compute(update_history=True)
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  # History: [2, 3, 4].
  expected = torch.stack(
    [
      torch.full((3,), 2.0, device=device),
      torch.full((3,), 3.0, device=device),
      torch.full((3,), 4.0, device=device),
    ]
  )
  assert torch.allclose(policy_obs[0], expected)

  # Update with value=5, circular overwrite.
  obs = manager.compute(update_history=True)
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  # History: [3, 4, 5].
  expected = torch.stack(
    [
      torch.full((3,), 3.0, device=device),
      torch.full((3,), 4.0, device=device),
      torch.full((3,), 5.0, device=device),
    ]
  )
  assert torch.allclose(policy_obs[0], expected)


def test_update_history_false_doesnt_modify_buffer(mock_env, simple_obs_func):
  """Test that update_history=False doesn't modify the buffer."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func, params={}, history_length=2, flatten_history_dim=False
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)

  # Initialize (value=1).
  obs1 = manager.compute(update_history=False)
  policy_obs1 = obs1["policy"]
  assert isinstance(policy_obs1, torch.Tensor)

  # Call without update (value=2, but buffer unchanged).
  obs2 = manager.compute(update_history=False)
  policy_obs2 = obs2["policy"]
  assert isinstance(policy_obs2, torch.Tensor)

  # History should still be [1, 1].
  assert torch.allclose(policy_obs1, policy_obs2)


# Group-level history override tests.


def test_group_history_overrides_term(mock_env, simple_obs_func):
  """Test group history_length overrides term history_length."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      history_length: int | None = 5  # Group level.
      flatten_history_dim: bool = False
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func,
          params={},
          history_length=2,  # Overridden.
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)
  assert manager.group_obs_dim["policy"] == (5, 3)

  obs = manager.compute()
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert policy_obs.shape == (4, 5, 3)


# History reset tests.


def test_reset_clears_all_envs(mock_env, simple_obs_func):
  """Test that reset without env_ids clears all environments."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func, params={}, history_length=2, flatten_history_dim=False
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)

  # Build up history.
  manager.compute(update_history=True)
  manager.compute(update_history=True)

  # Reset all.
  manager.reset()

  # Buffer should be zeroed.
  obs = manager.compute(update_history=False)
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert torch.allclose(policy_obs, torch.zeros((4, 2, 3), device=mock_env.device))


def test_reset_partial_envs(mock_env, simple_obs_func):
  """Test that reset with specific env_ids only resets those envs."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func, params={}, history_length=3, flatten_history_dim=False
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)

  # Build up history.
  manager.compute(update_history=True)
  manager.compute(update_history=True)
  obs_before = manager.compute(update_history=True)
  policy_obs_before = obs_before["policy"]
  assert isinstance(policy_obs_before, torch.Tensor)

  # Reset only envs 0 and 2.
  manager.reset(env_ids=torch.tensor([0, 2], device=mock_env.device))

  obs_after = manager.compute(update_history=False)
  policy_obs_after = obs_after["policy"]
  assert isinstance(policy_obs_after, torch.Tensor)

  # Envs 0 and 2 should be reset (zeros), 1 and 3 unchanged.
  assert torch.allclose(
    policy_obs_after[0], torch.zeros((3, 3), device=mock_env.device)
  )
  assert torch.allclose(policy_obs_after[1], policy_obs_before[1])
  assert torch.allclose(
    policy_obs_after[2], torch.zeros((3, 3), device=mock_env.device)
  )
  assert torch.allclose(policy_obs_after[3], policy_obs_before[3])


def test_reset_partial_envs_with_backfill(mock_env, simple_obs_func):
  """Test that reset envs get backfilled on next update."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func, params={}, history_length=3, flatten_history_dim=False
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)
  device = mock_env.device

  # Build history: [2, 3, 4] for all envs (counter starts at 1 after _prepare_terms).
  manager.compute(update_history=True)  # value=2
  manager.compute(update_history=True)  # value=3
  manager.compute(update_history=True)  # value=4

  # Reset only envs 0 and 2.
  manager.reset(env_ids=torch.tensor([0, 2], device=device))

  # Next update with value=5.
  obs = manager.compute(update_history=True)
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)

  # Env 0: [5, 5, 5] (backfilled after reset).
  expected_env0 = torch.stack(
    [
      torch.full((3,), 5.0, device=device),
      torch.full((3,), 5.0, device=device),
      torch.full((3,), 5.0, device=device),
    ]
  )
  assert torch.allclose(policy_obs[0], expected_env0)

  # Env 1: [3, 4, 5] (continues normally).
  expected_env1 = torch.stack(
    [
      torch.full((3,), 3.0, device=device),
      torch.full((3,), 4.0, device=device),
      torch.full((3,), 5.0, device=device),
    ]
  )
  assert torch.allclose(policy_obs[1], expected_env1)

  # Env 2: [5, 5, 5] (backfilled after reset).
  assert torch.allclose(policy_obs[2], expected_env0)

  # Env 3: [3, 4, 5] (continues normally).
  assert torch.allclose(policy_obs[3], expected_env1)


# History with other features tests.


def test_history_with_clip(mock_env, simple_obs_func):
  """Test that clipping is applied before history."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func,
          params={},
          history_length=2,
          flatten_history_dim=False,
          clip=(-0.5, 0.5),
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)

  obs = manager.compute(update_history=True)
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  # Values should be clipped.
  assert torch.all(policy_obs >= -0.5)
  assert torch.all(policy_obs <= 0.5)


def test_history_with_scale(mock_env, simple_obs_func):
  """Test that scaling is applied before history."""

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs1: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func,
          params={},
          history_length=2,
          flatten_history_dim=False,
          scale=2.0,
        )
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)
  device = mock_env.device
  # Counter at 1 after _prepare_terms.

  # First call uses value=2, scaled to 4, initializes buffer.
  manager.compute(update_history=False)
  # Second call uses value=3, scaled to 6, and updates history.
  obs = manager.compute(update_history=True)
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)

  # History: [4, 6].
  expected = torch.stack(
    [torch.full((3,), 4.0, device=device), torch.full((3,), 6.0, device=device)]
  )
  assert torch.allclose(policy_obs[0], expected)


# Mixed history terms tests.


def test_mixed_terms_concatenated(mock_env, simple_obs_func, device):
  """Test group with both history and non-history terms concatenated."""

  counter = {"value": 0}

  def obs_func2(env):
    counter["value"] += 1
    return torch.full((env.num_envs, 2), float(counter["value"]) * 10, device=device)

  @dataclass
  class ObsCfg:
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
      obs_with_history: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(
          func=simple_obs_func, params={}, history_length=2, flatten_history_dim=True
        )
      )
      obs_no_history: ObservationTermCfg = field(
        default_factory=lambda: ObservationTermCfg(func=obs_func2, params={})
      )

    policy: PolicyCfg = field(default_factory=PolicyCfg)

  manager = ObservationManager(ObsCfg(), mock_env)

  # Should concatenate: (3*2) + 2 = 8.
  assert manager.group_obs_dim["policy"] == (8,)

  obs = manager.compute()
  policy_obs = obs["policy"]
  assert isinstance(policy_obs, torch.Tensor)
  assert policy_obs.shape == (4, 8)
