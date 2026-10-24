"""Checkpoint publication and training lifecycle regressions."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, checkpoint
from mjlab.scripts import train
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg


def test_failed_save_keeps_previous_checkpoint(tmp_path, monkeypatch):
  destination = tmp_path / "model_1.pt"
  checkpoint.save_checkpoint({"value": torch.tensor([1])}, destination)
  original = destination.read_bytes()

  def interrupted_save(state, temporary):
    Path(temporary).write_bytes(b"partial ZIP")
    assert destination.read_bytes() == original
    raise OSError("disk full")

  monkeypatch.setattr(torch, "save", interrupted_save)
  with pytest.raises(OSError, match="disk full"):
    checkpoint.save_checkpoint({"value": torch.tensor([2])}, destination)
  assert destination.read_bytes() == original
  assert list(tmp_path.iterdir()) == [destination]


def test_checkpoint_reader_sees_complete_files(tmp_path, monkeypatch):
  destination = tmp_path / "model_1.pt"
  checkpoint.save_checkpoint({"value": 1}, destination)
  real_save = torch.save

  def inspected_save(state, temporary):
    assert torch.load(destination, weights_only=True)["value"] == 1
    real_save(state, temporary)
    assert torch.load(destination, weights_only=True)["value"] == 1

  monkeypatch.setattr(torch, "save", inspected_save)
  checkpoint.save_checkpoint({"value": 2}, destination)
  assert torch.load(destination, weights_only=True)["value"] == 2


@pytest.mark.parametrize("failures", [2, 5])
def test_checkpoint_replacement_retries_access_denied(tmp_path, monkeypatch, failures):
  destination = tmp_path / "model_1.pt"
  checkpoint.save_checkpoint({"value": 1}, destination)
  replace = checkpoint.os.replace
  attempts = []

  def replace_with_reader(source, target):
    attempts.append(target)
    if len(attempts) <= failures:
      raise PermissionError("reader holds file")
    replace(source, target)

  monkeypatch.setattr(checkpoint.os, "replace", replace_with_reader)
  monkeypatch.setattr(checkpoint.time, "sleep", lambda _: None)
  if failures == 5:
    with pytest.raises(PermissionError, match="reader holds file"):
      checkpoint.save_checkpoint({"value": 2}, destination)
    expected = 1
  else:
    checkpoint.save_checkpoint({"value": 2}, destination)
    expected = 2
  assert torch.load(destination, weights_only=True)["value"] == expected
  assert list(tmp_path.iterdir()) == [destination]


def test_missing_resume_checkpoint_does_not_allocate_env(tmp_path, monkeypatch):
  allocate = MagicMock()
  monkeypatch.setattr(train, "ManagerBasedRlEnv", allocate)
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  cfg = train.TrainConfig(
    env=cartpole_balance_env_cfg(), agent=RslRlOnPolicyRunnerCfg(resume=True)
  )
  with pytest.raises(ValueError, match="No run directories"):
    train.run_train("test", cfg, tmp_path / "new_run")
  allocate.assert_not_called()


def test_training_failure_closes_environment(tmp_path, monkeypatch):
  env = MagicMock()
  runner = MagicMock()
  runner.learn.side_effect = RuntimeError("rollout failed")
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train, "ManagerBasedRlEnv", lambda **kw: env)
  monkeypatch.setattr(train, "RslRlVecEnvWrapper", lambda env, **kw: env)
  monkeypatch.setattr(train, "load_runner_cls", lambda task: lambda *a, **kw: runner)
  monkeypatch.setattr(train, "dump_yaml", lambda *a: None)
  monkeypatch.setattr(train, "add_wandb_tags", lambda *a: None)
  cfg = train.TrainConfig(
    env=cartpole_balance_env_cfg(), agent=RslRlOnPolicyRunnerCfg()
  )
  with pytest.raises(RuntimeError, match="rollout failed"):
    train.run_train("test", cfg, tmp_path)
  env.close.assert_called_once()


def test_runner_failure_closes_logging_writer(monkeypatch):
  runner = object.__new__(MjlabOnPolicyRunner)
  runner._stateful_resume = False
  runner.logger = MagicMock()
  writer = runner.logger.writer

  def fail(*args):
    raise RuntimeError("learner failed")

  monkeypatch.setattr(train.MjlabOnPolicyRunner.__bases__[0], "learn", fail)
  with pytest.raises(RuntimeError, match="learner failed"):
    runner.learn(1)
  writer.close.assert_called_once()
  assert runner.logger.writer is None
