"""Tests for CurriculumManager."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.curriculum_manager import CurriculumManager, CurriculumTermCfg


@pytest.fixture
def mock_env():
  env = Mock()
  env.num_envs = 2
  return env


def test_get_active_iterable_terms_handles_dict_and_scalar_state(mock_env):
  """Dict- and scalar-shaped curriculum states both yield flat value lists.

  Regression: the dict branch previously indexed `terms` (a list) by term
  name, raising TypeError. Only observable through callers that invoke
  get_active_iterable_terms, which no in-tree caller currently does.
  """

  def dict_state_func(env, env_ids):
    return {"a": torch.tensor(1.5), "b": 2.0}

  def scalar_state_func(env, env_ids):
    return torch.tensor(7.0)

  cfg = {
    "dict_term": CurriculumTermCfg(func=dict_state_func, params={}),
    "scalar_term": CurriculumTermCfg(func=scalar_state_func, params={}),
  }
  manager = CurriculumManager(cfg, mock_env)
  manager.compute()

  terms = dict(manager.get_active_iterable_terms(0))
  assert terms["dict_term"] == [1.5, 2.0]
  assert terms["scalar_term"] == [7.0]
