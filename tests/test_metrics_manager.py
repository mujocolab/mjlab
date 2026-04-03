"""Tests for metrics manager functionality."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.metrics_manager import (
  MetricsManager,
  MetricsTermCfg,
  NullMetricsManager,
)


class SimpleTestMetric:
  """A class-based metric that tracks state."""

  def __init__(self, cfg: MetricsTermCfg, env):
    self.call_count = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env, **kwargs):
    self.call_count += 1
    return torch.ones(env.num_envs, device=env.device) * 0.5

  def reset(self, env_ids: torch.Tensor | None = None, env=None):
    if env_ids is not None and len(env_ids) > 0:
      self.call_count[env_ids] = 0


@pytest.fixture
def mock_env():
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.scene = {"robot": Mock()}
  return env


def test_episode_averages_and_reset(mock_env):
  """Compute for N steps, reset a subset, verify averages and zeroing."""
  cfg = {
    "term": MetricsTermCfg(
      func=lambda env: torch.ones(env.num_envs, device=env.device) * 0.5,
      params={},
    )
  }
  manager = MetricsManager(cfg, mock_env)

  for _ in range(10):
    manager.compute()

  info = manager.reset(env_ids=torch.tensor([0, 1]))

  # Each env: sum=5.0, count=10, avg=0.5. Mean across 2 reset envs = 0.5.
  assert info["Episode_Metrics/term"].item() == pytest.approx(0.5)
  # Reset envs zeroed; non-reset envs untouched.
  assert manager._episode_sums["term"][0] == 0.0
  assert manager._step_count[0] == 0
  assert manager._episode_sums["term"][2] == pytest.approx(5.0)
  assert manager._step_count[2] == 10


def test_early_termination_uses_per_env_step_count(mock_env):
  """Envs with different episode lengths get correct per-step averages."""
  step = [0]

  def step_dependent_metric(env):
    step[0] += 1
    return torch.full((env.num_envs,), float(step[0]), device=env.device)

  cfg = {"m": MetricsTermCfg(func=step_dependent_metric, params={})}
  manager = MetricsManager(cfg, mock_env)

  # 4 steps for all envs: values are 1, 2, 3, 4.
  for _ in range(4):
    manager.compute()
  # Env 0: sum=10, count=4. Reset it (env 1 keeps accumulating).
  manager.reset(env_ids=torch.tensor([0]))

  # 2 more steps: values are 5, 6.
  for _ in range(2):
    manager.compute()
  # Env 0: sum=11, count=2, avg=5.5.
  # Env 1: sum=21, count=6, avg=3.5.
  info = manager.reset(env_ids=torch.tensor([0, 1]))
  # Mean of [5.5, 3.5] = 4.5.
  assert info["Episode_Metrics/m"].item() == pytest.approx(4.5)


def test_class_based_metric_reset_targets_correct_envs(mock_env):
  """Class-based term's reset() is called with the correct env_ids."""
  cfg = {"term": MetricsTermCfg(func=SimpleTestMetric, params={})}
  manager = MetricsManager(cfg, mock_env)
  term = manager._class_term_cfgs[0].func

  for _ in range(10):
    manager.compute()

  manager.reset(env_ids=torch.tensor([0, 2]))

  assert term.call_count[0] == 0
  assert term.call_count[1] == 10
  assert term.call_count[2] == 0
  assert term.call_count[3] == 10


def test_null_metrics_manager(mock_env):
  """NullMetricsManager doesn't crash and returns empty dict on reset."""
  manager = NullMetricsManager()
  manager.compute()
  assert manager.reset(env_ids=torch.tensor([0])) == {}


def test_none_terms_are_skipped(mock_env):
  """None terms in config are skipped without error."""
  cfg: dict[str, MetricsTermCfg | None] = {
    "valid": MetricsTermCfg(
      func=lambda env: torch.ones(env.num_envs, device=env.device),
      params={},
    ),
    "skipped": None,
  }
  manager = MetricsManager(cfg, mock_env)  # type: ignore[arg-type]
  assert manager._term_names == ["valid"]


def test_per_substep_averaging(mock_env):
  """Substep terms are averaged across substeps; step terms evaluated once."""
  substep_call = [0]

  def rising_metric(env):
    substep_call[0] += 1
    return torch.full((env.num_envs,), float(substep_call[0]), device=env.device)

  cfg = {
    "step_term": MetricsTermCfg(
      func=lambda env: torch.ones(env.num_envs, device=env.device) * 2.0,
      params={},
    ),
    "substep_term": MetricsTermCfg(func=rising_metric, params={}, per_substep=True),
  }
  manager = MetricsManager(cfg, mock_env)

  for _ in range(2):  # 2 env steps
    for _ in range(4):  # 4 substeps each
      manager.compute_substep()
    manager.compute()

  info = manager.reset(env_ids=torch.tensor([0]))
  assert info["Episode_Metrics/step_term"].item() == pytest.approx(2.0)
  # Substeps: step 0 -> [1,2,3,4] avg=2.5; step 1 -> [5,6,7,8] avg=6.5.
  # Episode avg = (2.5 + 6.5) / 2 = 4.5.  A broken single-eval path
  # would give (1 + 2) / 2 = 1.5.
  assert info["Episode_Metrics/substep_term"].item() == pytest.approx(4.5)


def test_substep_reset_zeroes_accumulators(mock_env):
  """Reset zeroes substep accumulators for the given env_ids."""
  cfg = {
    "sub": MetricsTermCfg(
      func=lambda env: torch.ones(env.num_envs, device=env.device),
      params={},
      per_substep=True,
    ),
  }
  manager = MetricsManager(cfg, mock_env)

  # Accumulate some substep values without calling compute().
  for _ in range(3):
    manager.compute_substep()

  # Reset env 0 -- its substep accum should be zeroed.
  manager.reset(env_ids=torch.tensor([0]))
  assert manager._substep_accum[0][0].item() == 0.0
  # Env 1 still has its accumulated value.
  assert manager._substep_accum[0][1].item() == pytest.approx(3.0)


def test_reduce_last_reports_final_step_value(mock_env):
  """reduce='last' reports the last step's value, not the episode average."""
  step = [0]

  def rising_metric(env):
    step[0] += 1
    return torch.full((env.num_envs,), float(step[0]), device=env.device)

  cfg = {"term": MetricsTermCfg(func=rising_metric, params={}, reduce="last")}
  manager = MetricsManager(cfg, mock_env)

  for _ in range(4):
    manager.compute()

  info = manager.reset(env_ids=torch.tensor([0]))

  # Values were 1, 2, 3, 4. Mean would be 2.5; last should be 4.
  assert info["Episode_Metrics/term"].item() == pytest.approx(4.0)


def test_reduce_mean_and_last_coexist(mock_env):
  """Mixed reduce modes in the same manager report correctly."""
  step = [0]

  def rising_metric(env):
    step[0] += 1
    return torch.full((env.num_envs,), float(step[0]), device=env.device)

  cfg = {
    "mean_term": MetricsTermCfg(func=rising_metric, params={}, reduce="mean"),
    "last_term": MetricsTermCfg(func=rising_metric, params={}, reduce="last"),
  }
  manager = MetricsManager(cfg, mock_env)

  for _ in range(3):
    manager.compute()

  info = manager.reset(env_ids=torch.tensor([0]))

  # mean_term sees steps 1, 3, 5 (odd calls); episode avg = 3.0.
  # last_term sees steps 2, 4, 6 (even calls); last value = 6.0.
  assert info["Episode_Metrics/mean_term"].item() == pytest.approx(3.0)
  assert info["Episode_Metrics/last_term"].item() == pytest.approx(6.0)


def test_no_substep_terms_no_overhead(mock_env):
  """When no per_substep terms exist, compute_substep is a no-op."""
  cfg = {
    "step_only": MetricsTermCfg(
      func=lambda env: torch.ones(env.num_envs, device=env.device),
      params={},
    ),
  }
  manager = MetricsManager(cfg, mock_env)

  # Should not raise or change state.
  manager.compute_substep()
  assert manager._substep_count == 0

  manager.compute()
  info = manager.reset(env_ids=torch.tensor([0]))
  assert info["Episode_Metrics/step_only"].item() == pytest.approx(1.0)
