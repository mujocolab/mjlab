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
