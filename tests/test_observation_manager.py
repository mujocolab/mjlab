"""Tests for ObservationManager behavior."""

from unittest.mock import Mock

import pytest
from conftest import get_test_device

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationManager,
)


@pytest.fixture
def mock_env():
  env = Mock()
  env.num_envs = 4
  env.device = get_test_device()
  env.step_dt = 0.02
  return env


def test_empty_terms_dict_raises(mock_env):
  """A group declared with no terms should fail fast at construction."""
  cfg = {"actor": ObservationGroupCfg(terms={})}

  with pytest.raises(ValueError, match="'actor' has no active terms"):
    ObservationManager(cfg, mock_env)


def test_all_terms_none_raises(mock_env):
  """A group whose every term is None should also fail fast."""
  cfg = {
    "actor": ObservationGroupCfg(
      terms={"a": None, "b": None},  # type: ignore[dict-item]
    ),
  }

  with pytest.raises(ValueError, match="'actor' has no active terms"):
    ObservationManager(cfg, mock_env)
