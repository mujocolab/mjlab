"""Integration tests for walking-to-football policy compatibility."""

import io
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import load_env_cfg, load_runner_cls
from mjlab.tasks.velocity_football.rl.runner import VelocityOnPolicyRunner

_PRETRAIN_TASK_ID = "Mjlab-Velocity-Football-Pretrain-Flat-Unitree-G1"
_FOOTBALL_TASK_ID = "Mjlab-Velocity-Football-Flat-Unitree-G1"


class _FakeActor:
  def __init__(self, state: dict[str, torch.Tensor]) -> None:
    self._state = {key: value.clone() for key, value in state.items()}

  def state_dict(self) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in self._state.items()}

  def load_state_dict(
    self, state: dict[str, torch.Tensor], strict: bool = True
  ) -> None:
    assert strict
    self._state = {key: value.clone() for key, value in state.items()}


def _actor_state(obs_dim: int, fill: float) -> dict[str, torch.Tensor]:
  return {
    "obs_normalizer._mean": torch.full((1, obs_dim), fill),
    "obs_normalizer._var": torch.full((1, obs_dim), fill + 1),
    "obs_normalizer._std": torch.full((1, obs_dim), fill + 2),
    "obs_normalizer.count": torch.tensor(123),
    "distribution.std_param": torch.full((29,), fill + 3),
    "mlp.0.weight": torch.full((4, obs_dim), fill + 4),
    "mlp.0.bias": torch.full((4,), fill + 5),
    "mlp.2.weight": torch.full((3, 4), fill + 6),
    "mlp.2.bias": torch.full((3,), fill + 7),
    "mlp.4.weight": torch.full((2, 3), fill + 8),
    "mlp.4.bias": torch.full((2,), fill + 9),
    "mlp.6.weight": torch.full((29, 2), fill + 10),
    "mlp.6.bias": torch.full((29,), fill + 11),
  }


def _fake_runner(actor: _FakeActor) -> VelocityOnPolicyRunner:
  runner = object.__new__(VelocityOnPolicyRunner)
  untyped_runner = cast(Any, runner)
  untyped_runner.alg = SimpleNamespace(actor=actor)
  untyped_runner.device = "cpu"
  untyped_runner.current_learning_iteration = 0
  untyped_runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
  return runner


def test_load_pretrained_transfers_only_walking_actor_prefix(tmp_path) -> None:
  source = _actor_state(490, fill=2.0)
  target_actor = _FakeActor(_actor_state(535, fill=10.0))
  runner = _fake_runner(target_actor)
  algorithm = cast(Any, runner.alg)
  algorithm.critic = object()
  algorithm.optimizer = object()
  critic = algorithm.critic
  optimizer = algorithm.optimizer

  checkpoint = tmp_path / "walking.pt"
  torch.save(
    {
      "actor_state_dict": source,
      "critic_state_dict": {"must_not_load": torch.tensor(1)},
      "optimizer_state_dict": {"must_not_load": True},
      "iter": 900,
      "infos": {"env_state": {"common_step_counter": 5000}},
    },
    checkpoint,
  )

  runner.load_pretrained(str(checkpoint))
  actual = target_actor.state_dict()

  for key, source_value in source.items():
    if key in VelocityOnPolicyRunner._NORMALIZER_VECTOR_KEYS:
      torch.testing.assert_close(actual[key][..., :490], source_value)
    elif key == VelocityOnPolicyRunner._FIRST_LAYER_KEY:
      torch.testing.assert_close(actual[key][:, :490], source_value)
    else:
      torch.testing.assert_close(actual[key], source_value)

  assert torch.count_nonzero(actual["mlp.0.weight"][:, 490:]) == 0
  torch.testing.assert_close(
    actual["obs_normalizer._mean"][:, 490:], torch.full((1, 45), 10.0)
  )
  torch.testing.assert_close(
    actual["obs_normalizer._var"][:, 490:], torch.full((1, 45), 11.0)
  )
  torch.testing.assert_close(
    actual["obs_normalizer._std"][:, 490:], torch.full((1, 45), 12.0)
  )
  assert algorithm.critic is critic
  assert algorithm.optimizer is optimizer
  assert runner.current_learning_iteration == 0
  assert runner.env.unwrapped.common_step_counter == 0


def test_load_pretrained_rejects_wrong_actor_observation_size(tmp_path) -> None:
  checkpoint = tmp_path / "native_velocity.pt"
  torch.save({"actor_state_dict": _actor_state(99, fill=1.0)}, checkpoint)
  runner = _fake_runner(_FakeActor(_actor_state(535, fill=10.0)))

  with pytest.raises(ValueError, match="walking Actor with 490 observations"):
    runner.load_pretrained(str(checkpoint))


def test_load_pretrained_rejects_incompatible_hidden_layer(tmp_path) -> None:
  source = _actor_state(490, fill=1.0)
  source["mlp.2.weight"] = torch.ones(5, 4)
  checkpoint = tmp_path / "wrong_hidden_layer.pt"
  torch.save({"actor_state_dict": source}, checkpoint)
  runner = _fake_runner(_FakeActor(_actor_state(535, fill=10.0)))

  with pytest.raises(ValueError, match="mlp.2.weight"):
    runner.load_pretrained(str(checkpoint))


def test_football_tasks_register_transfer_runner() -> None:
  assert load_runner_cls(_PRETRAIN_TASK_ID) is VelocityOnPolicyRunner
  assert load_runner_cls(_FOOTBALL_TASK_ID) is VelocityOnPolicyRunner


def test_pretrained_checkpoint_is_mutually_exclusive_with_resume(tmp_path) -> None:
  cfg = TrainConfig.from_task(_FOOTBALL_TASK_ID)
  cfg.agent.resume = True
  cfg = replace(cfg, pretrained_checkpoint=str(tmp_path / "walking.pt"))

  with pytest.raises(ValueError, match="cannot be combined"):
    run_train(_FOOTBALL_TASK_ID, cfg, tmp_path / "logs")


def test_pretrained_checkpoint_must_exist(tmp_path) -> None:
  cfg = replace(
    TrainConfig.from_task(_FOOTBALL_TASK_ID),
    pretrained_checkpoint=str(tmp_path / "missing.pt"),
  )

  with pytest.raises(FileNotFoundError, match="Pretrained checkpoint not found"):
    run_train(_FOOTBALL_TASK_ID, cfg, tmp_path / "logs")


@pytest.mark.slow
def test_pretrain_observations_are_strict_football_prefix() -> None:
  environments = []
  try:
    for task_id in (_PRETRAIN_TASK_ID, _FOOTBALL_TASK_ID):
      cfg = load_env_cfg(task_id)
      cfg.scene.num_envs = 1
      output = io.StringIO()
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(output), redirect_stderr(output):
          environments.append(ManagerBasedRlEnv(cfg, device="cpu"))

    pretrain, football = environments
    pretrain_obs, _ = pretrain.reset(seed=7)
    football_obs, _ = football.reset(seed=7)

    assert pretrain_obs["actor"].shape == (1, 490)
    assert pretrain_obs["critic"].shape == (1, 505)
    assert football_obs["actor"].shape == (1, 535)
    assert football_obs["critic"].shape == (1, 565)
    assert pretrain.action_space.shape == football.action_space.shape == (1, 29)

    pretrain_terms = pretrain.observation_manager.active_terms
    football_terms = football.observation_manager.active_terms
    assert (
      football_terms["actor"][: len(pretrain_terms["actor"])] == pretrain_terms["actor"]
    )
    assert (
      football_terms["critic"][: len(pretrain_terms["critic"])]
      == pretrain_terms["critic"]
    )
  finally:
    for env in environments:
      env.close()
