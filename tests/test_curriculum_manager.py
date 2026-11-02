"""Tests for CurriculumManager."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.curriculum_manager import (
  CurriculumManager,
  CurriculumTermCfg,
  NullCurriculumManager,
)


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


class StageCurriculum:
  """A curriculum whose progress cannot be reconstructed from the step count."""

  def __init__(self, cfg, env):
    self.stage = 0

  def __call__(self, env, env_ids):
    return self.stage

  def state_dict(self):
    return {"stage": self.stage}

  def load_state_dict(self, state):
    self.stage = state["stage"]


def _stateless_curriculum(env, env_ids):
  return 7


@pytest.fixture
def checkpoint_manager(mock_env):
  return CurriculumManager(
    {
      "first": CurriculumTermCfg(func=StageCurriculum),
      "second": CurriculumTermCfg(func=StageCurriculum),
      "stateless": CurriculumTermCfg(func=_stateless_curriculum),
    },
    mock_env,
  )


def test_curriculum_checkpoint_round_trip(checkpoint_manager, mock_env, tmp_path):
  checkpoint_manager.get_term_cfg("first").func.stage = 3
  checkpoint_manager.get_term_cfg("second").func.stage = 5
  state = checkpoint_manager.state_dict()
  assert state == {"first": {"stage": 3}, "second": {"stage": 5}}
  path = tmp_path / "curriculum.pt"
  torch.save(state, path)

  restored = CurriculumManager(
    {
      "first": CurriculumTermCfg(func=StageCurriculum),
      "second": CurriculumTermCfg(func=StageCurriculum),
    },
    mock_env,
  )
  restored.load_state_dict(torch.load(path, weights_only=True))
  restored.compute()
  assert dict(restored.get_active_iterable_terms(0)) == {
    "first": [3],
    "second": [5],
  }


def test_curriculum_checkpoint_missing_and_unknown_terms(checkpoint_manager):
  checkpoint_manager.get_term_cfg("second").func.stage = 8
  checkpoint_manager.load_state_dict(
    {
      "first": {"stage": 4},
      "removed": {"stage": 9},
      "stateless": {"stage": 2},
    }
  )
  assert checkpoint_manager.get_term_cfg("first").func.stage == 4
  assert checkpoint_manager.get_term_cfg("second").func.stage == 8


def test_stateless_curriculum_checkpoint(mock_env):
  manager = CurriculumManager(
    {"stateless": CurriculumTermCfg(func=_stateless_curriculum)}, mock_env
  )
  manager.compute()
  assert manager.state_dict() == {}


def test_empty_curriculum_checkpoint(mock_env):
  for manager in (CurriculumManager({}, mock_env), NullCurriculumManager()):
    assert manager.state_dict() == {}
    manager.load_state_dict({"removed": {"stage": 1}})
