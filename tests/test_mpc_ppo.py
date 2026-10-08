"""Tests for MPC data collection (stage 2) and MPC-guided PPO (stage 3)."""

from dataclasses import asdict, replace

import pytest
import torch
from conftest import get_test_device

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mpopi_train.algorithms import MpopiCfg, MpopiPpo
from mpopi_train.algorithms.config import MpcDataCfg
from mpopi_train.config import with_mpopi
from mpopi_train.mpc import SamplingMpcCfg
from mpopi_train.mpc.collector import MpcCollector
from mpopi_train.runner import MpopiOnPolicyRunner

TASK = "Mjlab-Cartpole-Balance"
TINY_PLANNER = SamplingMpcCfg(num_samples=4, horizon=3)


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _runner(
  device: str, mpc: MpcDataCfg, seed: int = 0, **ppo_overrides
) -> MpopiOnPolicyRunner:
  env_cfg = load_env_cfg(TASK)
  env_cfg.scene.num_envs = 8
  env_cfg.seed = seed
  agent = load_rl_cfg(TASK)
  assert isinstance(agent, RslRlOnPolicyRunnerCfg)
  agent.num_steps_per_env = 8
  agent.seed = seed
  agent.logger = "tensorboard"
  agent.algorithm = replace(agent.algorithm, **ppo_overrides)
  mpopi_agent = with_mpopi(agent, MpopiCfg(mode="mpc_ppo", mpc=mpc))
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent.clip_actions
  )
  return MpopiOnPolicyRunner(env, asdict(mpopi_agent), log_dir=None, device=device)


def _alg(runner: MpopiOnPolicyRunner) -> MpopiPpo:
  assert isinstance(runner.alg, MpopiPpo)
  return runner.alg


def _close(runner: MpopiOnPolicyRunner) -> None:
  collector = _alg(runner).mpc_collector
  assert collector is not None
  collector.close()
  runner.env.close()


def _learn(runner: MpopiOnPolicyRunner, iterations: int) -> list[dict]:
  logs: list[dict] = []
  runner.logger.log = lambda **kw: logs.append(kw["loss_dict"])  # type: ignore[method-assign]
  runner.learn(num_learning_iterations=iterations)
  return logs


def test_mpc_data_cfg_schedule_and_round_trip():
  cfg = MpcDataCfg(collect_every=2, collect_iterations=5, bc_coef=2.0, bc_iterations=4)
  assert [cfg.collects(i) for i in range(7)] == [1, 0, 1, 0, 1, 0, 0]
  assert [cfg.bc_weight(i) for i in (0, 2, 4, 9)] == [2.0, 1.0, 0.0, 0.0]
  full = MpopiCfg(mode="mpc_ppo", mpc=replace(cfg, planner=TINY_PLANNER))
  assert MpopiCfg.from_dict(asdict(full)) == full
  with pytest.raises(ValueError, match="behavior density"):
    MpopiCfg(mode="mpc_ppo", mpc=MpcDataCfg(execution_std=0.0)).validate()
  with pytest.raises(ValueError, match="DAgger labels"):
    MpopiCfg(mode="mpc_ppo", mpc=MpcDataCfg(driver="policy")).validate()
  # Naive injection needs no density.
  MpopiCfg(
    mode="mpc_ppo", mpc=MpcDataCfg(execution_std=0.0, correction=False)
  ).validate()
  with pytest.raises(ValueError, match="inject_fraction"):
    MpopiCfg(mode="mpc_ppo", mpc=MpcDataCfg(inject_fraction=1.0)).validate()
  floored = replace(cfg, bc_floor=0.5)
  assert [floored.bc_weight(i) for i in (0, 2, 4, 9)] == [2.0, 1.0, 0.5, 0.5]


def test_collector_records_exact_behavior_density(device):
  collector = MpcCollector(
    load_env_cfg(TASK),
    num_envs=3,
    num_steps=5,
    planner_cfg=TINY_PLANNER,
    execution_std=0.3,
    device=device,
  )
  try:
    segment, metrics = collector.collect()
  finally:
    collector.close()
  mean, std = segment["behavior_distribution_params"]
  assert segment["observations"].batch_size == torch.Size([5, 3])
  assert segment["actions"].shape == mean.shape == (5, 3, 1)
  assert segment["bootstrap_observations"].batch_size == torch.Size([3])
  for key in ("rewards", "dones", "time_outs", "behavior_log_prob"):
    assert segment[key].shape == (5, 3, 1), key
  assert (std == 0.3).all()
  # MPC actions respect the planner's clip; executed actions add the noise.
  assert (mean.abs() <= TINY_PLANNER.action_clip + 1e-6).all()  # type: ignore[operator]
  expected = torch.distributions.Normal(mean, std).log_prob(segment["actions"])
  torch.testing.assert_close(segment["behavior_log_prob"], expected.sum(-1, True))
  assert metrics["seconds"] > 0.0


def test_mpc_ppo_uses_mpc_data_then_becomes_ppo(device):
  mpc = MpcDataCfg(
    num_envs=2,
    num_steps=4,
    collect_iterations=2,
    buffer_segments=4,
    max_age=2,
    bc_iterations=2,
    planner=TINY_PLANNER,
  )
  runner = _runner(device, mpc)
  logs = _learn(runner, 5)
  collected = ["mpc/collect_reward" in log for log in logs]
  assert collected == [True, True, False, False, False]
  assert logs[0]["mpc/bc_loss"] > 0.0 and logs[0]["mpc/bc_weight"] == 1.0
  assert logs[2]["mpc/bc_weight"] == 0.0
  assert logs[1]["mpopi/accepted"] == 2 * 2 * 4  # Both segments, all accepted.
  # Segments from iterations 0 and 1 age out after max_age = 2.
  assert [log["mpc/buffer_segments"] for log in logs] == [1, 2, 2, 1, 0]
  assert "mpopi/accepted" not in logs[4]  # Plain PPO once the buffer is empty.
  for p in runner.alg.actor.parameters():
    assert torch.isfinite(p).all()
  _close(runner)


def test_behavior_cloning_pulls_policy_toward_mpc(device):
  mpc = MpcDataCfg(
    num_envs=4,
    num_steps=8,
    collect_iterations=1,
    max_age=None,
    use_in_ppo=False,
    bc_coef=10.0,
    bc_iterations=100,
    planner=TINY_PLANNER,
  )
  errors = []
  for bc_coef in (0.0, mpc.bc_coef):
    # Fixed learning rate: the adaptive KL schedule throttles large BC steps.
    runner = _runner(device, replace(mpc, bc_coef=bc_coef), schedule="fixed")
    _learn(runner, 10)
    alg = _alg(runner)
    buffer = alg.replay
    assert buffer.observations is not None
    obs = buffer.observations[0].flatten(0, 1)
    target = buffer.behavior_distribution_params[0][0].flatten(0, 1)
    with torch.no_grad():
      alg.actor(obs, stochastic_output=True)
      mean = alg.actor.output_distribution_params[0]
    errors.append(float((mean - target).square().mean()))
    _close(runner)
  # Same seed, same MPC data: only the BC term differs.
  assert errors[1] < 0.7 * errors[0]


def test_collector_labels_without_noise_and_for_dagger(device):
  collector = MpcCollector(
    load_env_cfg(TASK),
    num_envs=2,
    num_steps=3,
    planner_cfg=TINY_PLANNER,
    execution_std=0.0,
    device=device,
  )
  try:
    plain, _ = collector.collect()
    dagger, _ = collector.collect(lambda obs: torch.full((2, 1), 0.7, device=device))
  finally:
    collector.close()
  # Without noise the MPC action itself is executed.
  torch.testing.assert_close(plain["actions"], plain["behavior_distribution_params"][0])
  # DAgger: the policy acts, the MPC action is only the label.
  assert (dagger["actions"] == 0.7).all()
  assert not (dagger["behavior_distribution_params"][0] == 0.7).all()
  for segment in (plain, dagger):
    assert (segment["behavior_log_prob"] == 0.0).all()  # No density.


def test_dagger_with_bc_floor_keeps_cloning_after_collection(device):
  mpc = MpcDataCfg(
    num_envs=2,
    num_steps=4,
    collect_iterations=2,
    buffer_segments=4,
    max_age=None,
    execution_std=0.0,
    driver="policy",
    use_in_ppo=False,
    bc_iterations=2,
    bc_floor=0.1,
    planner=TINY_PLANNER,
  )
  runner = _runner(device, mpc)
  logs = _learn(runner, 4)
  assert [log["mpc/bc_weight"] for log in logs] == [1.0, 0.5, 0.1, 0.1]
  assert [log["mpc/buffer_segments"] for log in logs] == [1, 2, 2, 2]
  assert logs[3]["mpc/bc_loss"] > 0.0  # Collection stopped; cloning goes on.
  assert all("mpopi/accepted" not in log for log in logs)  # Not in PPO's loss.
  for p in _alg(runner).actor.parameters():
    assert torch.isfinite(p).all()
  _close(runner)


def test_dagger_with_own_replay_uses_both_sources(device):
  mpc = MpcDataCfg(
    num_envs=2,
    num_steps=4,
    collect_iterations=3,
    execution_std=0.0,
    driver="policy",
    use_in_ppo=False,
    bc_iterations=3,
    replay_own_rollouts=True,
    planner=TINY_PLANNER,
  )
  with pytest.raises(ValueError, match="replay_own_rollouts"):
    replace(mpc, driver="mpc", execution_std=0.3, use_in_ppo=True).validate()
  runner = _runner(device, replace(mpc, teacher_gap_every=2))
  alg = _alg(runner)
  logs = _learn(runner, 3)
  assert alg.own_replay is not None and len(alg.own_replay) == 3
  # Teacher-versus-policy comparison on 2 of the 4 collected steps.
  for log in logs:
    assert 0.0 <= log["mpc/collect_teacher_better_frac"] <= 1.0
    assert log["mpc/collect_teacher_gap"] == pytest.approx(
      log["mpc/collect_teacher_return"] - log["mpc/collect_teacher_policy_return"],
      abs=1e-5,
    )
  assert len(alg.replay) == 3  # MPC labels, in their own buffer.
  num_fresh = 8 * 8
  for log in logs[1:]:
    assert log["mpopi/accepted"] > 0  # Past PPO data enters PPO's loss...
    # ...but the MPC labels do not: only fresh and replayed PPO samples count.
    assert log["mpopi/gradient_samples"] == num_fresh + log["mpopi/accepted"]
    assert log["mpc/bc_loss"] > 0.0
  for p in alg.actor.parameters():
    assert torch.isfinite(p).all()
  _close(runner)


def test_mpc_injection_mixes_a_fixed_fraction_for_the_whole_run(device):
  mpc = MpcDataCfg(
    num_envs=2,
    num_steps=4,
    collect_iterations=2,
    buffer_segments=4,
    max_age=None,
    execution_std=0.0,
    correction=False,
    bc_coef=0.0,
    inject_fraction=0.2,
    planner=TINY_PLANNER,
  )
  runner = _runner(device, mpc)
  logs = _learn(runner, 4)
  num_fresh = 8 * 8  # num_envs x num_steps_per_env of the PPO runner.
  # 1 segment (8 samples) after the first collection, then 16 samples kept.
  assert [log["mpopi/sampled"] for log in logs] == [8, 16, 16, 16]
  # Once enough data exists, MPC is 0.2 of the batch: 16 / (64 + 16).
  assert logs[3]["mpopi/sampled"] / (num_fresh + logs[3]["mpopi/sampled"]) == 0.2
  assert all(log["mpopi/accepted"] == log["mpopi/sampled"] for log in logs)
  assert all(log["mpopi/weight_mean"] == 1.0 for log in logs)  # Naive: w = 1.
  for p in _alg(runner).actor.parameters():
    assert torch.isfinite(p).all()
  _close(runner)


@pytest.mark.parametrize("mode", ["ppo", "mpc_ppo"])
def test_min_action_std_bounds_the_actor_std(device, mode):
  env_cfg = load_env_cfg(TASK)
  env_cfg.scene.num_envs = 4
  agent = load_rl_cfg(TASK)
  assert isinstance(agent, RslRlOnPolicyRunnerCfg)
  agent.logger = "tensorboard"
  mpc = MpcDataCfg(num_envs=1, num_steps=2, planner=TINY_PLANNER)
  mpopi = MpopiCfg(mode=mode, mpc=mpc, min_action_std=0.3)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
  runner = MpopiOnPolicyRunner(
    env, asdict(with_mpopi(agent, mpopi)), log_dir=None, device=device
  )
  actor = runner.alg.actor
  with torch.no_grad():
    actor.distribution.std_param.fill_(-1.0)  # type: ignore[union-attr]
    actor(env.get_observations(), stochastic_output=True)
  torch.testing.assert_close(
    actor.output_distribution_params[1], torch.full((4, 1), 0.3, device=device)
  )
  if mode == "mpc_ppo":
    _close(runner)
  else:
    env.close()
