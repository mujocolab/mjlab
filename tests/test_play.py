"""Checkpoint selection in play without starting simulation or a viewer."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import mjlab.scripts.play as play
from mjlab.rl import RslRlOnPolicyRunnerCfg


@pytest.fixture
def play_runtime(monkeypatch):
  runner = Mock()
  viewer = Mock()
  monkeypatch.setattr(play, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(
    play, "load_env_cfg", lambda *args, **kwargs: SimpleNamespace(commands={})
  )
  monkeypatch.setattr(
    play,
    "load_rl_cfg",
    lambda task: RslRlOnPolicyRunnerCfg(experiment_name="test_velocity"),
  )
  monkeypatch.setattr(play, "ManagerBasedRlEnv", Mock())
  monkeypatch.setattr(play, "RslRlVecEnvWrapper", lambda env, **kwargs: env)
  monkeypatch.setattr(play, "load_runner_cls", lambda task: Mock(return_value=runner))
  monkeypatch.setattr(play, "ViserPlayViewer", viewer)
  return runner, viewer


def test_play_latest_checkpoint(tmp_path: Path, play_runtime, capsys):
  """Select by run timestamp then iteration, and keep Viser tied to that run."""
  log_root = tmp_path / "custom_logs"
  experiment = log_root / "test_velocity"
  run = experiment / "2026-11-10_12-00-00"
  expected = run / "model_1000.pt"
  for checkpoint in (
    experiment / "2026-11-09_12-00-00" / "model_5000.pt",
    run / "model_900.pt",
    expected,
    experiment / "wandb_checkpoints" / "model_9000.pt",
    log_root / "other_task" / "2026-11-11_12-00-00" / "model_9000.pt",
  ):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
  (run / "params").mkdir()

  play.run_play(
    "Test-Task",
    play.PlayConfig(
      checkpoint_file="latest", log_root=str(log_root), device="cpu", viewer="viser"
    ),
  )

  runner, viewer = play_runtime
  runner.load.assert_called_once_with(
    str(expected), load_cfg={"actor": True}, strict=True, map_location="cpu"
  )
  assert str(expected) in capsys.readouterr().out
  manager = viewer.call_args.kwargs["checkpoint_manager"]
  assert manager.current_name == "model_1000.pt"
  newest = run / "model_1100.pt"
  newest.touch()
  assert manager.fetch_available()[-1][0] == newest.name
  manager.load_checkpoint(newest.name)
  runner.load.assert_called_with(
    str(newest), load_cfg={"actor": True}, strict=True, map_location="cpu"
  )


def test_play_explicit_checkpoint(tmp_path: Path, play_runtime):
  checkpoint = tmp_path / "model_100.pt"
  checkpoint.touch()

  play.run_play(
    "Test-Task",
    play.PlayConfig(checkpoint_file=str(checkpoint), device="cpu", viewer="viser"),
  )

  runner, _ = play_runtime
  runner.load.assert_called_once_with(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu"
  )


def test_play_latest_empty_run(tmp_path: Path, play_runtime):
  experiment = tmp_path / "test_velocity"
  older_run = experiment / "2026-11-09_12-00-00"
  older_run.mkdir(parents=True)
  (older_run / "model_100.pt").touch()
  (experiment / "2026-11-10_12-00-00").mkdir()

  with pytest.raises(ValueError, match="No checkpoint found"):
    play.run_play(
      "Test-Task",
      play.PlayConfig(
        checkpoint_file="latest", log_root=str(tmp_path), device="cpu", viewer="viser"
      ),
    )

  runner, viewer = play_runtime
  runner.load.assert_not_called()
  viewer.assert_not_called()
