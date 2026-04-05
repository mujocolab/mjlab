"""Tests for reward_curriculum and termination_curriculum."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.envs.mdp.curriculums import reward_curriculum, termination_curriculum
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg


def _reward_func(env):
  return torch.ones(env.num_envs)


def _termination_func(env):
  return torch.zeros(env.num_envs, dtype=torch.bool)


def _make_reward_cfg(
  weight: float = 1.0,
  params: dict | None = None,
) -> RewardTermCfg:
  return RewardTermCfg(
    func=_reward_func,
    weight=weight,
    params=params if params is not None else {"std": 0.5, "scale": 1.0},
  )


def _make_termination_cfg(
  params: dict | None = None,
) -> TerminationTermCfg:
  return TerminationTermCfg(
    func=_termination_func,
    params=params if params is not None else {"threshold": float("inf")},
  )


def _build_reward(env, reward_name, stages):
  params = {"reward_name": reward_name, "stages": stages}
  cfg = CurriculumTermCfg(func=reward_curriculum, params=params)
  instance = reward_curriculum(cfg, env)
  return instance(env, env_ids=torch.tensor([0, 1]), **params)


def _build_termination(env, termination_name, stages):
  params = {"termination_name": termination_name, "stages": stages}
  cfg = CurriculumTermCfg(func=termination_curriculum, params=params)
  instance = termination_curriculum(cfg, env)
  return instance(env, env_ids=torch.tensor([0, 1]), **params)


def _make_reward_env(step_counter, reward_cfg):
  env = Mock()
  env.common_step_counter = step_counter
  env.reward_manager.get_term_cfg.return_value = reward_cfg
  return env


def _make_termination_env(step_counter, term_cfg):
  env = Mock()
  env.common_step_counter = step_counter
  env.termination_manager.get_term_cfg.return_value = term_cfg
  return env


# Reward: weight


def test_reward_weight_unchanged_before_threshold():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  _build_reward(env, "r", [{"step": 100, "weight": 2.0}])
  assert rc.weight == pytest.approx(1.0)


def test_reward_weight_applied_at_threshold():
  rc = _make_reward_cfg()
  env = _make_reward_env(100, rc)
  _build_reward(env, "r", [{"step": 100, "weight": 2.0}])
  assert rc.weight == pytest.approx(2.0)


def test_reward_weight_later_stage_wins():
  rc = _make_reward_cfg()
  env = _make_reward_env(500, rc)
  _build_reward(
    env,
    "r",
    [
      {"step": 0, "weight": 0.5},
      {"step": 100, "weight": 1.5},
      {"step": 400, "weight": 3.0},
    ],
  )
  assert rc.weight == pytest.approx(3.0)


def test_reward_weight_partial_application():
  rc = _make_reward_cfg()
  env = _make_reward_env(150, rc)
  _build_reward(
    env,
    "r",
    [
      {"step": 100, "weight": 2.0},
      {"step": 200, "weight": 4.0},
    ],
  )
  assert rc.weight == pytest.approx(2.0)


def test_step_zero_applies_immediately():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  _build_reward(env, "r", [{"step": 0, "weight": 9.0}])
  assert rc.weight == pytest.approx(9.0)


# Reward: params


def test_reward_params_updated():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2}}])
  assert rc.params["std"] == 0.2


def test_reward_params_unchanged_before_threshold():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2}}])
  assert rc.params["std"] == 0.5


def test_reward_multiple_params_updated():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2, "scale": 2.0}}])
  assert rc.params["std"] == 0.2
  assert rc.params["scale"] == 2.0


# Reward: combined weight + params


def test_reward_weight_and_params_in_same_stage():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(env, "r", [{"step": 100, "weight": 5.0, "params": {"std": 0.1}}])
  assert rc.weight == pytest.approx(5.0)
  assert rc.params["std"] == 0.1


# Termination: params


def test_termination_params_updated():
  tc = _make_termination_cfg()
  env = _make_termination_env(200, tc)
  _build_termination(env, "energy", [{"step": 100, "params": {"threshold": 500.0}}])
  assert tc.params["threshold"] == 500.0


def test_termination_params_unchanged_before_threshold():
  tc = _make_termination_cfg()
  env = _make_termination_env(0, tc)
  _build_termination(env, "energy", [{"step": 100, "params": {"threshold": 500.0}}])
  assert tc.params["threshold"] == float("inf")


def test_termination_later_stage_wins():
  tc = _make_termination_cfg()
  env = _make_termination_env(500, tc)
  _build_termination(
    env,
    "energy",
    [
      {"step": 0, "params": {"threshold": 1000.0}},
      {"step": 100, "params": {"threshold": 700.0}},
      {"step": 400, "params": {"threshold": 400.0}},
    ],
  )
  assert tc.params["threshold"] == 400.0


# Validation: shared engine


def test_unknown_reward_param_raises():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  params = {"reward_name": "r", "stages": [{"step": 0, "params": {"stdd": 0.2}}]}
  cfg = CurriculumTermCfg(func=reward_curriculum, params=params)
  with pytest.raises(KeyError, match="unknown param"):
    reward_curriculum(cfg, env)


def test_unknown_termination_param_raises():
  tc = _make_termination_cfg()
  env = _make_termination_env(0, tc)
  params = {
    "termination_name": "energy",
    "stages": [{"step": 0, "params": {"thresholddd": 1.0}}],
  }
  cfg = CurriculumTermCfg(func=termination_curriculum, params=params)
  with pytest.raises(KeyError, match="unknown param"):
    termination_curriculum(cfg, env)


def test_unsorted_stages_raise():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  params = {
    "reward_name": "r",
    "stages": [
      {"step": 200, "weight": 1.0},
      {"step": 100, "weight": 2.0},
    ],
  }
  cfg = CurriculumTermCfg(func=reward_curriculum, params=params)
  with pytest.raises(ValueError, match="nondecreasing"):
    reward_curriculum(cfg, env)


def test_duplicate_steps_allowed():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(
    env,
    "r",
    [
      {"step": 100, "weight": 2.0},
      {"step": 100, "params": {"std": 0.1}},
    ],
  )
  assert rc.weight == pytest.approx(2.0)
  assert rc.params["std"] == 0.1


# Logging keys


def test_reward_logs_only_staged_keys():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  result = _build_reward(
    env, "r", [{"step": 100, "weight": 5.0, "params": {"std": 0.2}}]
  )
  assert result["weight"].item() == pytest.approx(5.0)
  assert result["std"].item() == pytest.approx(0.2)
  assert "scale" not in result  # Not in any stage.


def test_reward_omits_weight_when_not_staged():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  result = _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2}}])
  assert "weight" not in result
  assert "std" in result


def test_termination_log_keys():
  tc = _make_termination_cfg()
  env = _make_termination_env(200, tc)
  result = _build_termination(
    env, "energy", [{"step": 100, "params": {"threshold": 500.0}}]
  )
  assert "threshold" in result
  assert result["threshold"].item() == pytest.approx(500.0)
  assert "weight" not in result  # No weight for termination.
