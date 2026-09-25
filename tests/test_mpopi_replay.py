"""Tests for the MPOPI replay buffer and replay batch construction."""

import math

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.rl.mpopi import Mpopi, MpopiCfg, ReplayBuffer
from mjlab.rl.mpopi.estimators import vtrace

NUM_STEPS, NUM_ENVS, OBS_DIM, ACT_DIM = 6, 4, 3, 2
GAMMA, LAM = 0.99, 0.95


def _obs(*batch: int, seed: int = 0) -> TensorDict:
  g = torch.Generator().manual_seed(seed)
  data = {
    "actor": torch.randn(*batch, OBS_DIM, generator=g),
    "critic": torch.randn(*batch, OBS_DIM, generator=g),
  }
  return TensorDict(data, batch_size=list(batch))


@pytest.fixture
def models():
  torch.manual_seed(0)
  obs = _obs(NUM_ENVS)
  groups = {"actor": ["actor"], "critic": ["critic"]}
  actor = MLPModel(
    obs,
    groups,
    "actor",
    ACT_DIM,
    hidden_dims=(16,),
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(16,))
  return actor, critic


def _collect_segment(actor, seed: int, done_prob: float = 0.2) -> dict:
  """Roll a fake segment where ``actor`` is the behavior policy."""
  g = torch.Generator().manual_seed(seed)
  obs = _obs(NUM_STEPS, NUM_ENVS, seed=seed)
  with torch.no_grad():
    actions = actor(obs.flatten(0, 1), stochastic_output=True)
    log_prob = actor.get_output_log_prob(actions)
    params = tuple(
      p.view(NUM_STEPS, NUM_ENVS, -1) for p in actor.output_distribution_params
    )
  dones = torch.rand(NUM_STEPS, NUM_ENVS, 1, generator=g) < done_prob
  time_outs = dones & (torch.rand(NUM_STEPS, NUM_ENVS, 1, generator=g) < 0.5)
  return dict(
    observations=obs,
    actions=actions.view(NUM_STEPS, NUM_ENVS, ACT_DIM),
    rewards=torch.randn(NUM_STEPS, NUM_ENVS, 1, generator=g),
    dones=dones,
    time_outs=time_outs,
    behavior_log_prob=log_prob.view(NUM_STEPS, NUM_ENVS, 1),
    behavior_distribution_params=params,
    bootstrap_observations=_obs(NUM_ENVS, seed=seed + 1000),
  )


def test_replay_buffer_fifo_and_versions(models):
  actor, _ = models
  buffer = ReplayBuffer(capacity=3)
  assert len(buffer) == 0 and buffer.num_transitions == 0
  segments = [_collect_segment(actor, seed=i) for i in range(5)]
  for version, seg in enumerate(segments):
    buffer.insert(**seg, policy_version=version)
  assert len(buffer) == 3
  assert buffer.num_transitions == 3 * NUM_STEPS * NUM_ENVS
  slots = buffer.slots(current_version=5)
  assert [int(buffer.policy_version[i]) for i in slots] == [2, 3, 4]
  # Content of the newest slot matches the last inserted segment.
  newest = slots[-1]
  torch.testing.assert_close(buffer.actions[newest], segments[4]["actions"])
  torch.testing.assert_close(
    buffer.observations[newest]["actor"], segments[4]["observations"]["actor"]
  )
  assert buffer.observations.batch_size == torch.Size([3, NUM_STEPS, NUM_ENVS])
  buffer.clear()
  assert len(buffer) == 0


def test_policy_version_filter(models):
  actor, _ = models
  buffer = ReplayBuffer(capacity=4)
  for version in range(4):
    buffer.insert(**_collect_segment(actor, seed=version), policy_version=version)
  kept = buffer.slots(current_version=4, max_age=2)
  assert [int(buffer.policy_version[i]) for i in kept] == [2, 3]


def _process(models, cfg: MpopiCfg, segments, version: int):
  actor, critic = models
  buffer = ReplayBuffer(capacity=cfg.replay_buffer_size)
  for v, seg in segments:
    buffer.insert(**seg, policy_version=v)
  return Mpopi(cfg, GAMMA, LAM).process(
    buffer, actor, critic, version=version, num_fresh=NUM_STEPS * NUM_ENVS
  )


def test_replay_from_current_policy_matches_on_policy_targets(models):
  # If mu == pi_old, the weights are exactly 1 and the V-trace targets equal
  # on-policy GAE computed with the current critic.
  actor, critic = models
  seg = _collect_segment(actor, seed=0)
  cfg = MpopiCfg(mode="mpopi_ppo", sampling_strategy="all")
  batch, metrics = _process(models, cfg, [(0, seg)], version=1)
  assert batch is not None
  assert len(batch) == NUM_STEPS * NUM_ENVS
  torch.testing.assert_close(batch.weights, torch.ones_like(batch.weights))
  assert batch.mask.all()
  assert metrics["ess"] == pytest.approx(1.0)
  assert metrics["behavior_kl"] == pytest.approx(0.0, abs=1e-6)

  with torch.no_grad():
    values = critic(seg["observations"].flatten(0, 1)).view(NUM_STEPS, NUM_ENVS, 1)
    boot = critic(seg["bootstrap_observations"])
  rewards = seg["rewards"] + GAMMA * values * seg["time_outs"]
  returns, adv = vtrace(
    rewards, seg["dones"], values, boot, torch.zeros_like(values), GAMMA, LAM
  )
  torch.testing.assert_close(batch.returns, returns.flatten(0, 1))
  torch.testing.assert_close(batch.advantages, adv.flatten(0, 1))
  torch.testing.assert_close(
    batch.old_actions_log_prob, seg["behavior_log_prob"].flatten(0, 1)
  )


@pytest.mark.parametrize(
  "clip_max, expected", [(None, 2.0), (1.0, 1.0), (1.5, 1.5)]
)
def test_known_ratio_is_weighted_and_clipped(models, clip_max, expected):
  actor, _ = models
  seg = _collect_segment(actor, seed=0)
  # mu(a|s) = pi_old(a|s) / 2  =>  rho = 2 everywhere.
  seg["behavior_log_prob"] = seg["behavior_log_prob"] - math.log(2.0)
  cfg = MpopiCfg(
    mode="mpopi_ppo",
    sampling_strategy="all",
    importance_weight_clip_max=clip_max,
  )
  batch, metrics = _process(models, cfg, [(0, seg)], version=1)
  assert batch is not None
  torch.testing.assert_close(
    batch.weights, torch.full_like(batch.weights, expected)
  )
  assert metrics["raw_ratio_mean"] == pytest.approx(2.0)
  assert metrics["clipped_frac"] == (0.0 if clip_max is None else 1.0)


def test_naive_mode_uses_unit_weights_and_same_selection(models):
  actor, _ = models
  seg = _collect_segment(actor, seed=0)
  seg["behavior_log_prob"] = seg["behavior_log_prob"] - math.log(2.0)
  seg["behavior_log_prob"][0, 0] = float("-inf")
  results = {}
  for mode in ("naive_replay_ppo", "mpopi_ppo"):
    torch.manual_seed(0)
    cfg = MpopiCfg(mode=mode, replay_ratio=0.5, importance_weight_clip_max=None)
    results[mode] = _process(models, cfg, [(0, seg)], version=1)[0]
  naive, corrected = results["naive_replay_ppo"], results["mpopi_ppo"]
  assert naive is not None and corrected is not None
  assert len(naive) == NUM_STEPS * NUM_ENVS // 2
  torch.testing.assert_close(naive.actions, corrected.actions)
  assert torch.equal(naive.mask, corrected.mask)
  torch.testing.assert_close(naive.weights, naive.mask.float())
  accepted = corrected.mask
  torch.testing.assert_close(
    corrected.weights[accepted], torch.full_like(corrected.weights[accepted], 2.0)
  )


def test_rejections_and_ess_gate(models):
  actor, _ = models
  seg = _collect_segment(actor, seed=0)
  seg["behavior_log_prob"][0, 0] = float("nan")
  seg["behavior_log_prob"][1, 0] -= 10.0  # |log rho| = 10.
  cfg = MpopiCfg(mode="mpopi_ppo", sampling_strategy="all", max_abs_log_ratio=5.0)
  batch, metrics = _process(models, cfg, [(0, seg)], version=1)
  assert batch is not None
  assert metrics["rejected_nonfinite"] == 1
  assert metrics["rejected_log_ratio"] == 1
  assert metrics["accepted"] == NUM_STEPS * NUM_ENVS - 2
  assert torch.isfinite(batch.weights).all()
  assert (batch.weights[~batch.mask] == 0).all()

  # A single dominant weight collapses ESS and trips the gate.
  seg["behavior_log_prob"][2, 0] -= 15.0
  cfg = MpopiCfg(
    mode="mpopi_ppo",
    sampling_strategy="all",
    importance_weight_clip_max=None,
    min_ess=0.5,
  )
  batch, metrics = _process(models, cfg, [(0, seg)], version=1)
  assert batch is not None
  assert metrics["ess"] < 0.5
  assert not batch.mask.any() and (batch.weights == 0).all()
  assert metrics["rejected_ess"] > 0


def test_age_filter_and_metrics(models):
  actor, _ = models
  segments = [(v, _collect_segment(actor, seed=v)) for v in range(3)]
  cfg = MpopiCfg(
    mode="mpopi_ppo",
    replay_buffer_size=3,
    sampling_strategy="all",
    max_policy_age=2,
  )
  batch, metrics = _process(models, cfg, segments, version=3)
  assert batch is not None
  assert len(batch) == 2 * NUM_STEPS * NUM_ENVS
  assert set(batch.policy_age.unique().tolist()) == {1, 2}
  assert metrics["stale_frac"] == pytest.approx(1 / 3)
  assert metrics["policy_age_max"] == 2


def test_empty_buffer_returns_none(models):
  cfg = MpopiCfg(mode="mpopi_ppo")
  batch, metrics = _process(models, cfg, [], version=0)
  assert batch is None
  assert metrics["buffer_segments"] == 0


def test_self_normalized_weights_have_unit_mean(models):
  actor, _ = models
  seg = _collect_segment(actor, seed=0)
  g = torch.Generator().manual_seed(1)
  seg["behavior_log_prob"] += torch.randn(seg["behavior_log_prob"].shape, generator=g)
  cfg = MpopiCfg(
    mode="mpopi_ppo",
    sampling_strategy="all",
    importance_weight_clip_max=None,
    weight_normalization="self_normalized",
  )
  batch, _ = _process(models, cfg, [(0, seg)], version=1)
  assert batch is not None
  assert batch.weights[batch.mask].mean().item() == pytest.approx(1.0, rel=1e-5)


def test_config_validation():
  with pytest.raises(ValueError):
    Mpopi(MpopiCfg(mode="ppo"), GAMMA, LAM)
  with pytest.raises(ValueError):
    MpopiCfg(mode="mpopi_ppo", max_policy_age=0).validate()
  with pytest.raises(ValueError):
    MpopiCfg(min_ess=1.5).validate()
