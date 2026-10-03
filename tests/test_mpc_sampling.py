"""Tests for the sampling MPC (MPPI / MPOPI) on mjlab environments."""

import math

import pytest
import torch
from conftest import get_test_device

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.mpc import SamplingMpc, SamplingMpcCfg, mppi_weights
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Cartpole-Balance"
NUM_REAL = 2


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def real_env(device):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = NUM_REAL
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  yield env
  env.close()


@pytest.fixture(scope="module")
def mpc(device):
  planner = SamplingMpc(
    load_env_cfg(TASK),
    NUM_REAL,
    SamplingMpcCfg(num_samples=8, horizon=5, seed=0),
    device=device,
  )
  yield planner
  planner.close()


def test_mppi_weights_hit_target_ess():
  g = torch.Generator().manual_seed(0)
  returns = torch.randn(4, 64, generator=g)
  weights, ess = mppi_weights(returns, target_ess=0.2)
  torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))
  torch.testing.assert_close(ess, torch.full((4,), 0.2), atol=1e-3, rtol=0)
  # Higher return always gets at least as much weight.
  order = returns.argsort(dim=1)
  sorted_w = weights.gather(1, order)
  assert (sorted_w[:, 1:] >= sorted_w[:, :-1] - 1e-7).all()


def test_mppi_weights_fixed_temperature_limits():
  returns = torch.tensor([[0.0, 1.0, 0.5]])
  greedy, ess = mppi_weights(returns, target_ess=None, temperature=1e-6)
  torch.testing.assert_close(greedy, torch.tensor([[0.0, 1.0, 0.0]]))
  assert ess.item() == pytest.approx(1 / 3)
  uniform, ess = mppi_weights(returns, target_ess=None, temperature=1e6)
  torch.testing.assert_close(uniform, torch.full((1, 3), 1 / 3), atol=1e-5, rtol=0)
  assert ess.item() == pytest.approx(1.0, abs=1e-5)


def test_state_copy_reproduces_real_env(real_env, mpc, device):
  # After copying, every planning world must follow the real env exactly.
  g = torch.Generator(device=device).manual_seed(1)
  for _ in range(3):
    real_env.step(torch.rand(NUM_REAL, 1, device=device, generator=g) * 2 - 1)
  mpc._copy_state(real_env)
  k = mpc.cfg.num_samples
  for _ in range(5):
    action = torch.rand(NUM_REAL, 1, device=device, generator=g) * 2 - 1
    real_env.step(action)
    mpc.env.step(action.repeat_interleave(k, dim=0))
    torch.testing.assert_close(
      mpc.env.sim.data.qpos.view(NUM_REAL, k, -1),
      real_env.sim.data.qpos[:, None].expand(-1, k, -1),
      atol=0.0,
      rtol=0.0,
    )
    torch.testing.assert_close(
      mpc.env.reward_buf.view(NUM_REAL, k),
      real_env.reward_buf[:, None].expand(-1, k),
      atol=0.0,
      rtol=0.0,
    )


def test_plan_outputs_and_warm_start(real_env, mpc):
  mpc.reset()
  plan = mpc.plan(real_env)
  assert plan.action.shape == (NUM_REAL, 1)
  assert plan.std.shape == (NUM_REAL, 1)
  assert (plan.action.abs() <= mpc.cfg.action_clip + 1e-6).all()
  assert torch.isfinite(plan.best_return).all()
  # The target ESS is clamped to the smallest achievable value, 1/K.
  expected_ess = max(mpc.cfg.target_ess or 0.0, 1.0 / mpc.cfg.num_samples)
  torch.testing.assert_close(
    plan.ess, torch.full_like(plan.ess, expected_ess), atol=0.01, rtol=0
  )
  # MPPI keeps the configured noise std; the warm-start plan repeats its tail.
  torch.testing.assert_close(plan.std, torch.full_like(plan.std, mpc.cfg.noise_std))
  torch.testing.assert_close(mpc.plan_seq[:, -1], mpc.plan_seq[:, -2])


def test_planning_does_not_touch_global_rng(real_env, mpc):
  before = torch.get_rng_state()
  mpc.plan(real_env)
  assert torch.equal(before, torch.get_rng_state())


def test_mpopi_adapts_std_within_bounds(real_env, device):
  cfg = SamplingMpcCfg(num_samples=8, horizon=5, iterations=3, seed=0)
  planner = SamplingMpc(load_env_cfg(TASK), NUM_REAL, cfg, device=device)
  try:
    plan = planner.plan(real_env)
  finally:
    planner.close()
  lo, hi = cfg.min_std_scale * cfg.noise_std, cfg.max_std_scale * cfg.noise_std
  assert (plan.std >= lo - 1e-6).all() and (plan.std <= hi + 1e-6).all()
  assert not torch.allclose(plan.std, torch.full_like(plan.std, cfg.noise_std))


def test_same_seed_gives_same_plan(real_env, device):
  plans = []
  for _ in range(2):
    planner = SamplingMpc(
      load_env_cfg(TASK),
      NUM_REAL,
      SamplingMpcCfg(num_samples=8, horizon=5, seed=3),
      device=device,
    )
    plans.append(planner.plan(real_env).action)
    planner.close()
  torch.testing.assert_close(plans[0], plans[1], atol=0.0, rtol=0.0)
  assert not any(math.isnan(v) for v in plans[0].flatten().tolist())


def test_knot_noise_is_piecewise_linear(device):
  cfg = SamplingMpcCfg(num_samples=3, horizon=7, num_knots=3, seed=0)
  planner = SamplingMpc(load_env_cfg(TASK), NUM_REAL, cfg, device=device)
  try:
    noise = planner._sample_noise(NUM_REAL, 3, 7, 1)
  finally:
    planner.close()
  assert noise.shape == (NUM_REAL, 3, 7, 1)
  # Knots at t = 0, 3, 6: second differences vanish inside each segment.
  second = noise[:, :, 2:] - 2 * noise[:, :, 1:-1] + noise[:, :, :-2]
  torch.testing.assert_close(
    second[:, :, [0, 1, 3, 4]],
    torch.zeros_like(second[:, :, [0, 1, 3, 4]]),
    atol=1e-5,
    rtol=0,
  )


def test_compare_with_policy_matches_open_loop_rollouts(real_env, mpc):
  mpc.plan(real_env)
  assert mpc.last_plan is not None
  n, k, h = NUM_REAL, mpc.cfg.num_samples, mpc.cfg.horizon

  def zero_policy(obs):
    return torch.zeros(obs.batch_size[0], mpc.action_dim, device=mpc.device)

  r_teacher, r_policy = mpc.compare_with_policy(real_env, zero_policy)
  # A policy that outputs zeros is the open-loop zero sequence; the teacher is
  # the open-loop plan. Both must match plain rollouts of those sequences.
  plan = mpc.last_plan[:, None].expand(n, k, h, -1).clone()
  torch.testing.assert_close(r_teacher, mpc._rollout(real_env, plan)[:, 0])
  zeros = torch.zeros_like(plan)
  torch.testing.assert_close(r_policy, mpc._rollout(real_env, zeros)[:, 0])
