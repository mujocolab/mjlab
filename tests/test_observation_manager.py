"""Tests for ObservationManager behavior."""

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
def mock_env():
  env = Mock()
  env.num_envs = 4
  env.device = get_test_device()
  env.step_dt = 0.02
  return env


def _dummy_term(mock_env):
  def func(_env, **_kwargs):
    return torch.zeros(mock_env.num_envs, 3, device=mock_env.device)

  return ObservationTermCfg(func=func)


def test_empty_terms_dict_skipped(mock_env):
  """A group declared with no terms is skipped rather than raising."""
  cfg = {"actor": ObservationGroupCfg(terms={})}
  mgr = ObservationManager(cfg, mock_env)

  assert "actor" not in mgr.active_terms
  assert "actor" not in mgr.group_obs_dim


def test_all_terms_none_skipped(mock_env):
  """A group whose every term is None is skipped rather than raising."""
  cfg = {
    "actor": ObservationGroupCfg(
      terms={"a": None, "b": None},  # type: ignore[dict-item]
    ),
  }
  mgr = ObservationManager(cfg, mock_env)

  assert "actor" not in mgr.active_terms
  assert "actor" not in mgr.group_obs_dim


def test_empty_group_skipped_alongside_active_group(mock_env):
  """Active groups coexist with empty ones; only the empty group is dropped."""
  cfg = {
    "actor": ObservationGroupCfg(terms={"a": _dummy_term(mock_env)}),
    "critic": ObservationGroupCfg(terms={}),
  }
  mgr = ObservationManager(cfg, mock_env)

  assert "actor" in mgr.active_terms
  assert "critic" not in mgr.active_terms


@pytest.mark.parametrize(
  "concatenate, dimension, flatten, history_b, features_b",
  [
    (True, -1, False, 2, 1),
    (True, 1, False, 2, 1),
    (True, 0, False, 3, 2),
    (True, -2, False, 3, 2),
    (False, -1, False, 2, 1),
    (True, -1, True, 3, 1),
  ],
)
def test_iterable_terms_preserve_term_values_with_history(
  mock_env, concatenate, dimension, flatten, history_b, features_b
):
  device = mock_env.device
  offsets = torch.arange(mock_env.num_envs, device=device).unsqueeze(1) * 10
  values = {
    "a": torch.tensor([[1.0, 2.0]], device=device) + offsets,
    "b": torch.arange(3.0, 3.0 + features_b, device=device).unsqueeze(0) + offsets,
  }
  initial = {name: value[2].tolist() for name, value in values.items()}
  manager = ObservationManager(
    {
      "actor": ObservationGroupCfg(
        terms={
          "a": ObservationTermCfg(
            func=lambda _env: values["a"],
            history_length=2,
            flatten_history_dim=flatten,
          ),
          "b": ObservationTermCfg(
            func=lambda _env: values["b"],
            history_length=history_b,
            flatten_history_dim=flatten,
          ),
        },
        concatenate_terms=concatenate,
        concatenate_dim=dimension,
      )
    },
    mock_env,
  )
  manager.compute(update_history=True)
  values = {name: value + 100 for name, value in values.items()}
  manager.compute(update_history=True)

  # Each term remains a flat sequence, oldest frame first, regardless of the
  # concatenation axis or whether history is flattened in the policy tensor.
  assert dict(manager.get_active_iterable_terms(2)) == {
    "actor-a": initial["a"] + [value + 100 for value in initial["a"]],
    "actor-b": initial["b"] * (history_b - 1) + [value + 100 for value in initial["b"]],
  }
