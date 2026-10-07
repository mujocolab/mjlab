"""Tests for the MPOPI -> PPO integration (MpopiPpo and the runner mode switch)."""

from dataclasses import asdict

import pytest
import torch
from rsl_rl.algorithms import PPO

from mjlab.rl import RslRlModelCfg
from mpopi_train.algorithms import MpopiCfg, MpopiPpo
from mpopi_train.algorithms.algorithm import weighted_clipped_surrogate
from mpopi_train.algorithms.toy_env import PointMassVecEnv
from mpopi_train.config import MpopiPpoAlgorithmCfg, MpopiRunnerCfg
from mpopi_train.runner import MpopiOnPolicyRunner

NUM_ENVS, NUM_STEPS = 16, 8


def _agent_cfg(mpopi: MpopiCfg | None = None) -> dict:
  cfg = MpopiRunnerCfg(
    num_steps_per_env=NUM_STEPS,
    actor=RslRlModelCfg(
      hidden_dims=(16,),
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    ),
    critic=RslRlModelCfg(hidden_dims=(16,)),
    algorithm=MpopiPpoAlgorithmCfg(
      num_learning_epochs=2,
      num_mini_batches=2,
      mpopi=mpopi or MpopiCfg(),
    ),
    logger="tensorboard",
  )
  return asdict(cfg)


def _runner(mpopi: MpopiCfg | None = None, seed: int = 0) -> MpopiOnPolicyRunner:
  torch.manual_seed(seed)
  env = PointMassVecEnv(num_envs=NUM_ENVS, max_episode_length=20, seed=seed)
  return MpopiOnPolicyRunner(env, _agent_cfg(mpopi), log_dir=None, device="cpu")


def _learn(runner: MpopiOnPolicyRunner, iters: int, monkeypatch) -> list[dict]:
  """Run ``learn`` and capture every iteration's loss dict."""
  logs: list[dict] = []
  monkeypatch.setattr(
    runner.logger, "log", lambda **kw: logs.append(dict(kw["loss_dict"]))
  )
  runner.learn(num_learning_iterations=iters)
  return logs


def test_ppo_mode_constructs_upstream_ppo():
  runner = _runner()
  assert type(runner.alg) is PPO
  assert "mpopi" not in runner.cfg["algorithm"]
  assert runner.cfg["algorithm"]["class_name"] == "PPO"


@pytest.mark.parametrize("mode", ["naive_replay_ppo", "mpopi_ppo"])
def test_replay_modes_construct_mpopi_ppo(mode):
  runner = _runner(MpopiCfg(mode=mode, replay_buffer_size=3, min_ess=0.2))
  assert isinstance(runner.alg, MpopiPpo)
  assert runner.alg.mpopi_cfg.mode == mode
  assert runner.alg.mpopi_cfg.replay_buffer_size == 3
  assert runner.alg.mpopi_cfg.min_ess == 0.2


def test_mpopi_rejects_non_ppo_algorithm():
  cfg = _agent_cfg(MpopiCfg(mode="mpopi_ppo"))
  cfg["algorithm"]["class_name"] = "Distillation"
  env = PointMassVecEnv(num_envs=NUM_ENVS)
  with pytest.raises(ValueError, match="requires the PPO algorithm"):
    MpopiOnPolicyRunner(env, cfg, log_dir=None, device="cpu")


@pytest.mark.parametrize(
  "mpopi, iters",
  [
    # The buffer is empty at the first update, so any replay mode equals PPO.
    (MpopiCfg(mode="mpopi_ppo"), 1),
    # With a zero replay size MPOPI never adds samples: PPO at every update.
    (MpopiCfg(mode="mpopi_ppo", replay_ratio=0.0), 3),
  ],
)
def test_no_replay_is_bit_identical_to_upstream_ppo(mpopi, iters, monkeypatch):
  ppo = _runner()
  mpopi_runner = _runner(mpopi)
  # Both runs must start from the same global RNG state (action sampling).
  torch.manual_seed(1)
  ppo_logs = _learn(ppo, iters, monkeypatch)
  torch.manual_seed(1)
  mpopi_logs = _learn(mpopi_runner, iters, monkeypatch)
  ppo_state = ppo.alg.save()
  mpopi_state = mpopi_runner.alg.save()
  for key in ("actor_state_dict", "critic_state_dict"):
    for name, tensor in ppo_state[key].items():
      assert torch.equal(tensor, mpopi_state[key][name]), name
  assert ppo.alg.learning_rate == mpopi_runner.alg.learning_rate
  for a, b in zip(ppo_logs, mpopi_logs, strict=True):
    for key in ("value", "surrogate", "entropy"):
      assert a[key] == b[key]


def test_surrogate_gradient_is_importance_weighted_policy_gradient():
  # At theta = theta_old (ratio 1, clipping inactive) the gradient of the MPOPI
  # surrogate must equal -mean(w * A * grad log pi).
  torch.manual_seed(0)
  n = 32
  mean = torch.randn(n, requires_grad=True)
  actions = torch.randn(n)
  dist = torch.distributions.Normal(mean, 1.0)
  log_prob = dist.log_prob(actions)
  advantages = torch.randn(n)
  weights = torch.rand(n) * 2
  loss, ratio = weighted_clipped_surrogate(
    log_prob, log_prob.detach(), advantages, clip_param=0.2, weights=weights
  )
  (grad,) = torch.autograd.grad(loss, mean)
  torch.testing.assert_close(ratio, torch.ones(n))
  grad_log_prob = actions - mean.detach()  # d/dmu log N(a; mu, 1)
  expected = -(weights * advantages * grad_log_prob) / n
  torch.testing.assert_close(grad, expected)


def test_surrogate_masks_and_weights():
  log_prob = torch.zeros(4)
  advantages = torch.tensor([1.0, 2.0, 3.0, 100.0])
  weights = torch.tensor([1.0, 0.5, 2.0, 0.0])
  mask = torch.tensor([True, True, True, False]).view(4, 1)
  loss, _ = weighted_clipped_surrogate(
    log_prob, log_prob, advantages, 0.2, weights=weights, mask=mask
  )
  # Rejected sample excluded from both numerator and count.
  assert loss.item() == pytest.approx(-(1.0 + 1.0 + 6.0) / 3)


@pytest.mark.parametrize("mode", ["naive_replay_ppo", "mpopi_ppo"])
def test_replay_training_runs_and_logs_metrics(mode, monkeypatch):
  cfg = MpopiCfg(mode=mode, replay_buffer_size=2, replay_ratio=1.0)
  runner = _runner(cfg)
  logs = _learn(runner, 4, monkeypatch)
  num_fresh = NUM_ENVS * NUM_STEPS
  assert logs[0]["mpopi/buffer_segments"] == 0
  assert logs[0]["mpopi/gradient_samples"] == num_fresh
  last = logs[-1]
  assert last["mpopi/buffer_segments"] == 2
  assert last["mpopi/sampled"] == num_fresh
  assert last["mpopi/accepted"] == num_fresh
  assert last["mpopi/gradient_samples"] == 2 * num_fresh
  assert 1 <= last["mpopi/policy_age_mean"] <= 2
  for key in ("kl", "clip_fraction", "mpopi/ess", "mpopi/behavior_kl"):
    assert key in last and torch.isfinite(torch.tensor(last[key]))
  if mode == "naive_replay_ppo":
    assert last["mpopi/weight_mean"] == 1.0 and last["mpopi/weight_std"] == 0.0
  else:
    assert last["mpopi/weight_max"] <= 1.0  # Default truncation at 1.
  for p in runner.alg.actor.parameters():
    assert torch.isfinite(p).all()


def test_extreme_behavior_log_probs_keep_training_finite(monkeypatch):
  cfg = MpopiCfg(
    mode="mpopi_ppo", replay_buffer_size=2, importance_weight_clip_max=None
  )
  runner = _runner(cfg)
  _learn(runner, 2, monkeypatch)
  alg = runner.alg
  assert isinstance(alg, MpopiPpo)
  blp = alg.replay.behavior_log_prob
  blp.view(-1)[0::3] = -1e6
  blp.view(-1)[1::3] = float("-inf")
  blp.view(-1)[2::7] = float("nan")
  logs = _learn(runner, 1, monkeypatch)
  assert logs[-1]["mpopi/rejected_nonfinite"] > 0
  for key, value in logs[-1].items():
    assert torch.isfinite(torch.tensor(value)), key
  for p in alg.actor.parameters():
    assert torch.isfinite(p).all()


def test_replay_stores_raw_rewards_and_time_outs(monkeypatch):
  runner = _runner(MpopiCfg(mode="mpopi_ppo", replay_buffer_size=1))
  _learn(runner, 3, monkeypatch)
  alg = runner.alg
  assert isinstance(alg, MpopiPpo)
  (slot,) = alg.replay.slots(alg.policy_version)
  assert int(alg.replay.policy_version[slot]) == alg.policy_version - 1
  # PPO's storage still holds the last rollout, with gamma * V(s_t) folded into
  # the rewards on time-out steps. The replay copy must hold the raw rewards.
  time_outs = alg.replay.time_outs[slot]
  assert time_outs.any()  # 20-step episodes, 24 collected steps.
  raw = alg.replay.rewards[slot]
  stored = alg.storage.rewards
  expected = raw + alg.gamma * alg.storage.values * time_outs
  torch.testing.assert_close(stored, expected)
  assert torch.equal(raw[~time_outs], stored[~time_outs])
