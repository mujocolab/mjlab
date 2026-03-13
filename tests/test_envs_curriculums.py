"""Tests for reward_curriculum."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg


@pytest.fixture
def mock_env():
  env = Mock()
  env.num_envs = 2
  env.device = "cpu"
  return env


@pytest.fixture
def reward_term_cfg():
  return RewardTermCfg(
    func=lambda env: torch.ones(env.num_envs),
    weight=1.0,
    params={"std": 0.5, "scale": 1.0},
  )


def _setup_env(env, step_counter, reward_term_cfg):
  env.common_step_counter = step_counter
  env.reward_manager.get_term_cfg.return_value = reward_term_cfg


def _call(env, **kwargs):
  return envs_mdp.reward_curriculum(env, env_ids=torch.tensor([0, 1]), **kwargs)


# Weight-only stages.


def test_weight_unchanged_before_threshold(mock_env, reward_term_cfg):
  _setup_env(mock_env, step_counter=0, reward_term_cfg=reward_term_cfg)
  _call(
    mock_env,
    reward_name="r",
    stages=[{"step": 100, "weight": 2.0}],
  )
  assert reward_term_cfg.weight == pytest.approx(1.0)


def test_weight_applied_at_threshold(mock_env, reward_term_cfg):
  """step: 100 applies when counter == 100 (>=, not >)."""
  _setup_env(mock_env, step_counter=100, reward_term_cfg=reward_term_cfg)
  _call(
    mock_env,
    reward_name="r",
    stages=[{"step": 100, "weight": 2.0}],
  )
  assert reward_term_cfg.weight == pytest.approx(2.0)


def test_weight_later_stage_wins(mock_env, reward_term_cfg):
  _setup_env(mock_env, step_counter=500, reward_term_cfg=reward_term_cfg)
  _call(
    mock_env,
    reward_name="r",
    stages=[
      {"step": 0, "weight": 0.5},
      {"step": 100, "weight": 1.5},
      {"step": 400, "weight": 3.0},
    ],
  )
  assert reward_term_cfg.weight == pytest.approx(3.0)


def test_weight_partial_application(mock_env, reward_term_cfg):
  _setup_env(mock_env, step_counter=150, reward_term_cfg=reward_term_cfg)
  _call(
    mock_env,
    reward_name="r",
    stages=[
      {"step": 100, "weight": 2.0},
      {"step": 200, "weight": 4.0},
    ],
  )
  assert reward_term_cfg.weight == pytest.approx(2.0)


# Params-only stages.


def test_params_updated(mock_env, reward_term_cfg):
  _setup_env(mock_env, step_counter=200, reward_term_cfg=reward_term_cfg)
  _call(
    mock_env,
    reward_name="r",
    stages=[{"step": 100, "params": {"std": 0.2}}],
  )
  assert reward_term_cfg.params["std"] == 0.2


def test_params_unchanged_before_threshold(mock_env, reward_term_cfg):
  _setup_env(mock_env, step_counter=0, reward_term_cfg=reward_term_cfg)
  _call(
    mock_env,
    reward_name="r",
    stages=[{"step": 100, "params": {"std": 0.2}}],
  )
  assert reward_term_cfg.params["std"] == 0.5


def test_multiple_params_updated(mock_env, reward_term_cfg):
  _setup_env(mock_env, step_counter=200, reward_term_cfg=reward_term_cfg)
  _call(
    mock_env,
    reward_name="r",
    stages=[{"step": 100, "params": {"std": 0.2, "scale": 2.0}}],
  )
  assert reward_term_cfg.params["std"] == 0.2
  assert reward_term_cfg.params["scale"] == 2.0


# Combined weight + params stages.


def test_weight_and_params_in_same_stage(mock_env, reward_term_cfg):
  _setup_env(mock_env, step_counter=200, reward_term_cfg=reward_term_cfg)
  _call(
    mock_env,
    reward_name="r",
    stages=[{"step": 100, "weight": 5.0, "params": {"std": 0.1}}],
  )
  assert reward_term_cfg.weight == pytest.approx(5.0)
  assert reward_term_cfg.params["std"] == 0.1


# Return value.


def test_return_includes_weight_and_scalar_params(mock_env, reward_term_cfg):
  _setup_env(mock_env, step_counter=200, reward_term_cfg=reward_term_cfg)
  result = _call(
    mock_env,
    reward_name="r",
    stages=[{"step": 100, "params": {"std": 0.2}}],
  )
  assert "weight" in result
  assert result["weight"].item() == pytest.approx(1.0)
  assert result["std"].item() == pytest.approx(0.2)


def test_unknown_param_raises(mock_env, reward_term_cfg):
  """Typos in param names are caught instead of silently ignored."""
  _setup_env(mock_env, step_counter=200, reward_term_cfg=reward_term_cfg)
  with pytest.raises(KeyError, match="unknown param"):
    _call(
      mock_env,
      reward_name="r",
      stages=[{"step": 100, "params": {"stdd": 0.2}}],
    )


def test_step_zero_applies_immediately(mock_env, reward_term_cfg):
  """A step: 0 stage applies on the very first call (counter == 0)."""
  _setup_env(mock_env, step_counter=0, reward_term_cfg=reward_term_cfg)
  _call(
    mock_env,
    reward_name="r",
    stages=[{"step": 0, "weight": 9.0}],
  )
  assert reward_term_cfg.weight == pytest.approx(9.0)
