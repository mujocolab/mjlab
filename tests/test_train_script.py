"""Tests for the train script's launch flow."""

import os
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mjlab.scripts.train as train_mod
import mjlab.tasks  # noqa: F401
from mjlab.scripts.train import TrainConfig, launch_training

TASK = "Mjlab-Velocity-Flat-Unitree-Go1"


def test_multi_gpu_resume_loads_previous_run(tmp_path, monkeypatch):
  """Resume must not pick the new run dir that torchrunx creates for its logs."""
  args = TrainConfig.from_task(TASK)
  args = replace(
    args,
    log_root=str(tmp_path),
    agent=replace(args.agent, resume=True, logger="tensorboard"),
  )
  previous = tmp_path / args.agent.experiment_name / "2000-01-01_00-00-00"
  previous.mkdir(parents=True)
  (previous / "model_100.pt").touch()

  # Two GPUs, and a launcher that creates TORCHRUNX_LOG_DIR before running the
  # workers, like torchrunx does.
  for var in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID"):
    monkeypatch.setenv(var, "")
  monkeypatch.delenv("TORCHRUNX_LOG_DIR", raising=False)
  monkeypatch.setattr(train_mod, "select_gpus", lambda gpu_ids: ([0, 1], 2))

  class FakeLauncher:
    def __init__(self, **kwargs):
      pass

    def run(self, fn, *fn_args):
      Path(os.environ["TORCHRUNX_LOG_DIR"]).mkdir(parents=True)
      fn(*fn_args)

  monkeypatch.setitem(
    sys.modules,
    "torchrunx",
    types.SimpleNamespace(Launcher=FakeLauncher, DEFAULT_ENV_VARS_FOR_COPY=()),
  )

  # Stub out the env and runner; the checkpoint lookup is what is under test.
  loaded = []

  class FakeRunner:
    def __init__(self, *args, **kwargs):
      pass

    def add_git_repo_to_log(self, path):
      pass

    def load(self, path):
      loaded.append(path)

    def learn(self, **kwargs):
      pass

  fake_env = SimpleNamespace(close=lambda: None)
  monkeypatch.setattr(train_mod, "ManagerBasedRlEnv", lambda **kwargs: fake_env)
  monkeypatch.setattr(train_mod, "RslRlVecEnvWrapper", lambda env, **kwargs: env)
  monkeypatch.setattr(train_mod, "load_runner_cls", lambda task_id: FakeRunner)

  launch_training(TASK, args)

  assert loaded == [str(previous / "model_100.pt")]
