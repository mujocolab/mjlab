"""Tests for reward curriculum functions (reward_weight, reward_params)."""

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


def test_no_stage_applied_before_threshold(mock_env, reward_term_cfg):
  """Params are unchanged when step counter hasn't passed any stage."""
  _setup_env(mock_env, step_counter=0, reward_term_cfg=reward_term_cfg)

  envs_mdp.reward_params(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    param_stages=[{"step": 100, "params": {"std": 0.2}}],
  )

  assert reward_term_cfg.params["std"] == 0.5


def test_stage_applied_after_threshold(mock_env, reward_term_cfg):
  """Params are updated when step counter exceeds a stage's threshold."""
  _setup_env(mock_env, step_counter=200, reward_term_cfg=reward_term_cfg)

  envs_mdp.reward_params(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    param_stages=[{"step": 100, "params": {"std": 0.2}}],
  )

  assert reward_term_cfg.params["std"] == 0.2


def test_later_stage_takes_precedence(mock_env, reward_term_cfg):
  """When multiple thresholds are exceeded, the last one wins."""
  _setup_env(mock_env, step_counter=500, reward_term_cfg=reward_term_cfg)

  envs_mdp.reward_params(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    param_stages=[
      {"step": 0, "params": {"std": 0.5}},
      {"step": 100, "params": {"std": 0.3}},
      {"step": 400, "params": {"std": 0.1}},
    ],
  )

  assert reward_term_cfg.params["std"] == 0.1


def test_partial_stage_application(mock_env, reward_term_cfg):
  """Only stages below the current step are applied; later ones are skipped."""
  _setup_env(mock_env, step_counter=150, reward_term_cfg=reward_term_cfg)

  envs_mdp.reward_params(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    param_stages=[
      {"step": 100, "params": {"std": 0.3}},
      {"step": 200, "params": {"std": 0.1}},
    ],
  )

  assert reward_term_cfg.params["std"] == 0.3


def test_multiple_params_updated(mock_env, reward_term_cfg):
  """Multiple param keys can be updated simultaneously."""
  _setup_env(mock_env, step_counter=200, reward_term_cfg=reward_term_cfg)

  envs_mdp.reward_params(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    param_stages=[{"step": 100, "params": {"std": 0.2, "scale": 2.0}}],
  )

  assert reward_term_cfg.params["std"] == 0.2
  assert reward_term_cfg.params["scale"] == 2.0


def test_return_value_contains_scalar_params(mock_env, reward_term_cfg):
  """Return dict includes scalar (int/float) params as tensors."""
  _setup_env(mock_env, step_counter=200, reward_term_cfg=reward_term_cfg)

  result = envs_mdp.reward_params(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    param_stages=[{"step": 100, "params": {"std": 0.2}}],
  )

  assert "std" in result
  assert isinstance(result["std"], torch.Tensor)
  assert result["std"].item() == pytest.approx(0.2)


def test_non_scalar_params_excluded_from_return(mock_env):
  """Non-numeric params (e.g. dicts) are excluded from the return value."""
  cfg = RewardTermCfg(
    func=lambda env: torch.ones(env.num_envs),
    weight=1.0,
    params={"std": {"joint_a": 0.3}, "scale": 1.0},
  )
  _setup_env(mock_env, step_counter=0, reward_term_cfg=cfg)

  result = envs_mdp.reward_params(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    param_stages=[],
  )

  assert "std" not in result
  assert "scale" in result


def test_dict_param_deep_merged(mock_env):
  """Dict-valued params are deep-merged so unrelated keys are preserved."""
  cfg = RewardTermCfg(
    func=lambda env: torch.ones(env.num_envs),
    weight=1.0,
    params={"std_walking": {".*knee.*": 0.5, ".*hip.*": 0.4}},
  )
  _setup_env(mock_env, step_counter=200, reward_term_cfg=cfg)

  envs_mdp.reward_params(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="pose",
    param_stages=[{"step": 100, "params": {"std_walking": {".*knee.*": 0.2}}}],
  )

  # Only the specified key is updated; the other is preserved.
  assert cfg.params["std_walking"][".*knee.*"] == pytest.approx(0.2)
  assert cfg.params["std_walking"][".*hip.*"] == pytest.approx(0.4)


# --- reward_weight ---


@pytest.fixture
def weight_term_cfg():
  return RewardTermCfg(
    func=lambda env: torch.ones(env.num_envs),
    weight=1.0,
    params={},
  )


def test_weight_unchanged_before_threshold(mock_env, weight_term_cfg):
  """Weight stays at initial value when no stage threshold is exceeded."""
  _setup_env(mock_env, step_counter=0, reward_term_cfg=weight_term_cfg)

  envs_mdp.reward_weight(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    weight_stages=[{"step": 100, "weight": 2.0}],
  )

  assert weight_term_cfg.weight == pytest.approx(1.0)


def test_weight_updated_after_threshold(mock_env, weight_term_cfg):
  """Weight is updated when step counter exceeds a stage's threshold."""
  _setup_env(mock_env, step_counter=200, reward_term_cfg=weight_term_cfg)

  envs_mdp.reward_weight(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    weight_stages=[{"step": 100, "weight": 2.0}],
  )

  assert weight_term_cfg.weight == pytest.approx(2.0)


def test_weight_later_stage_takes_precedence(mock_env, weight_term_cfg):
  """When multiple thresholds are exceeded, the last one wins."""
  _setup_env(mock_env, step_counter=500, reward_term_cfg=weight_term_cfg)

  envs_mdp.reward_weight(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    weight_stages=[
      {"step": 0, "weight": 0.5},
      {"step": 100, "weight": 1.5},
      {"step": 400, "weight": 3.0},
    ],
  )

  assert weight_term_cfg.weight == pytest.approx(3.0)


def test_weight_partial_stage_application(mock_env, weight_term_cfg):
  """Stages whose threshold hasn't been reached yet are not applied."""
  _setup_env(mock_env, step_counter=150, reward_term_cfg=weight_term_cfg)

  result = envs_mdp.reward_weight(
    mock_env,
    env_ids=torch.tensor([0, 1]),
    reward_name="lin_vel",
    weight_stages=[
      {"step": 100, "weight": 2.0},
      {"step": 200, "weight": 4.0},
    ],
  )

  assert isinstance(result, torch.Tensor)
  assert weight_term_cfg.weight == pytest.approx(2.0)
