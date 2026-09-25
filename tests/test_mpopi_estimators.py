"""Tests for MPOPI importance-sampling and V-trace estimators."""

import math

import pytest
import torch

from mjlab.rl.mpopi.estimators import (
  bootstrap_time_outs,
  effective_sample_size,
  importance_log_ratio,
  importance_weights,
  vtrace,
)


def _reference_gae(rewards, dones, values, last_values, gamma, lam):
  """Verbatim port of rsl_rl 5.5.1 PPO.compute_returns (before normalization)."""
  num_steps = rewards.shape[0]
  returns = torch.zeros_like(values)
  advantage = 0
  for step in reversed(range(num_steps)):
    next_values = last_values if step == num_steps - 1 else values[step + 1]
    next_is_not_terminal = 1.0 - dones[step].float()
    delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
    advantage = delta + next_is_not_terminal * gamma * lam * advantage
    returns[step] = advantage + values[step]
  return returns, returns - values


def _random_segment(num_steps=16, num_envs=5, seed=0):
  g = torch.Generator().manual_seed(seed)
  rewards = torch.randn(num_steps, num_envs, 1, generator=g)
  dones = torch.rand(num_steps, num_envs, 1, generator=g) < 0.15
  values = torch.randn(num_steps, num_envs, 1, generator=g)
  last_values = torch.randn(num_envs, 1, generator=g)
  return rewards, dones, values, last_values


def test_importance_ratio_from_probabilities():
  log_ratio = importance_log_ratio(
    torch.tensor(math.log(0.8)), torch.tensor(math.log(0.4))
  )
  weights, _ = importance_weights(log_ratio, clip_max=None)
  assert torch.isclose(weights, torch.tensor(2.0))


def test_weight_clipping():
  log_ratio = torch.log(torch.tensor([0.05, 0.5, 1.0, 3.0]))
  weights, clipped = importance_weights(log_ratio, clip_min=0.1, clip_max=1.0)
  torch.testing.assert_close(weights, torch.tensor([0.1, 0.5, 1.0, 1.0]))
  assert clipped.tolist() == [True, False, False, True]


def test_zero_and_near_zero_behavior_probability():
  target = torch.tensor([0.0, 0.0, 0.0])
  behavior = torch.tensor([-1e4, float("-inf"), float("nan")])
  log_ratio = importance_log_ratio(target, behavior)
  weights, _ = importance_weights(log_ratio, clip_max=None, log_ratio_clamp=20.0)
  assert torch.isfinite(weights).all()
  # Huge-but-finite ratio is clamped numerically, non-finite ones get weight 0.
  assert torch.isclose(weights[0], torch.tensor(math.exp(20.0)))
  assert weights[1] == 0.0 and weights[2] == 0.0


def test_log_domain_matches_float64_reference():
  g = torch.Generator().manual_seed(0)
  target = torch.randn(1000, generator=g) * 5 - 300.0
  behavior = target + torch.randn(1000, generator=g) * 3
  weights, _ = importance_weights(
    importance_log_ratio(target, behavior), clip_max=None
  )
  expected = torch.exp(target.double() - behavior.double())
  # Naive probability division underflows to 0/0 here; the log domain does not.
  assert torch.isnan(torch.exp(target) / torch.exp(behavior)).all()
  torch.testing.assert_close(weights.double(), expected, rtol=1e-4, atol=0.0)


def test_effective_sample_size():
  assert torch.isclose(effective_sample_size(torch.ones(10)), torch.tensor(1.0))
  one_hot = torch.zeros(10)
  one_hot[3] = 5.0
  assert torch.isclose(effective_sample_size(one_hot), torch.tensor(0.1))
  assert effective_sample_size(torch.zeros(4)) == 0.0
  assert effective_sample_size(torch.zeros(0)) == 0.0


def test_bootstrap_time_outs():
  rewards = torch.tensor([[1.0], [2.0]])
  time_outs = torch.tensor([[False], [True]])
  values = torch.tensor([[10.0], [10.0]])
  out = bootstrap_time_outs(rewards, time_outs, values, gamma=0.5)
  torch.testing.assert_close(out, torch.tensor([[1.0], [7.0]]))


@pytest.mark.parametrize("lam", [0.0, 0.95, 1.0])
def test_vtrace_on_policy_equals_rsl_rl_gae(lam):
  rewards, dones, values, last_values = _random_segment()
  expected_returns, expected_adv = _reference_gae(
    rewards, dones, values, last_values, 0.99, lam
  )
  returns, adv = vtrace(
    rewards,
    dones,
    values,
    last_values,
    torch.zeros_like(values),
    gamma=0.99,
    lam=lam,
  )
  torch.testing.assert_close(returns, expected_returns)
  torch.testing.assert_close(adv, expected_adv)


def test_vtrace_single_step_weighting():
  # One step, terminal: v = V + min(rho_bar, rho) * delta, A = delta.
  rewards = torch.tensor([[[1.0]]])
  dones = torch.ones(1, 1, 1, dtype=torch.bool)
  values = torch.zeros(1, 1, 1)
  log_ratio = torch.full((1, 1, 1), math.log(3.0))
  returns, adv = vtrace(
    rewards, dones, values, torch.zeros(1, 1), log_ratio, 0.9, 0.9, rho_clip=2.0
  )
  assert torch.isclose(adv, torch.tensor(1.0)).all()
  assert torch.isclose(returns, torch.tensor(2.0)).all()


def test_vtrace_cuts_trace_after_unlikely_action():
  # rho_1 = 0 at step 1: the target for step 0 must not see rewards past step 1.
  rewards = torch.tensor([0.0, 0.0, 100.0]).view(3, 1, 1)
  dones = torch.zeros(3, 1, 1, dtype=torch.bool)
  values = torch.zeros(3, 1, 1)
  log_ratio = torch.tensor([0.0, float("-inf"), 0.0]).view(3, 1, 1)
  returns, adv = vtrace(rewards, dones, values, torch.zeros(1, 1), log_ratio, 1.0, 1.0)
  assert adv[0].item() == 0.0 and returns[0].item() == 0.0
  assert adv[1].item() == 100.0  # A_1 conditions on a_1 and sees later rewards.
  assert returns[2].item() == 100.0


def test_extreme_ratios_stay_finite():
  rewards, dones, values, last_values = _random_segment(seed=1)
  log_ratio = torch.randn_like(values) * 1e4
  log_ratio[0, 0] = float("nan")
  log_ratio[1, 0] = float("inf")
  returns, adv = vtrace(
    rewards, dones, values, last_values, log_ratio, 0.99, 0.95, rho_clip=None
  )
  weights, _ = importance_weights(log_ratio, clip_max=None)
  for t in (returns, adv, weights):
    assert torch.isfinite(t).all()


def test_importance_weighting_recovers_target_expectation():
  # mu = N(0, 1), pi = N(1, 1), reward = a. Naive replay estimates E_mu[a] = 0,
  # importance weighting recovers E_pi[a] = 1.
  g = torch.Generator().manual_seed(0)
  mu, pi = torch.distributions.Normal(0.0, 1.0), torch.distributions.Normal(1.0, 1.0)
  actions = torch.randn(200_000, generator=g)  # Samples from mu.
  log_ratio = importance_log_ratio(pi.log_prob(actions), mu.log_prob(actions))
  weights, _ = importance_weights(log_ratio, clip_max=None)
  assert abs(actions.mean().item()) < 0.02
  assert abs((weights * actions).mean().item() - 1.0) < 0.05
  # Truncation at 1 biases the estimate back toward the behavior policy.
  truncated, _ = importance_weights(log_ratio, clip_max=1.0)
  estimate = (truncated * actions).mean().item()
  assert 0.0 < estimate < 1.0
