"""Typed options reach native RSL collection and compilation."""

from dataclasses import asdict

import pytest
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg


@pytest.mark.parametrize("configured", [False, True])
def test_native_performance_options_reach_collection_and_model_compilation(
  monkeypatch, configured
):
  class TensorEnv(VecEnv):
    num_envs = 2
    num_actions = 1
    device = "cpu"
    cfg = {}

    def get_observations(self):
      return TensorDict({"actor": torch.zeros(2, 3), "critic": torch.zeros(2, 3)}, [2])

    def step(self, actions):
      return self.get_observations(), torch.ones(2), torch.zeros(2), {}

  compile_modes = []
  checks = []

  def compile_model(model, mode):
    compile_modes.append(mode)
    return model

  monkeypatch.setattr("rsl_rl.algorithms.ppo.compile_model", compile_model)
  monkeypatch.setattr(
    "rsl_rl.runners.on_policy_runner.check_nan", lambda *a: checks.append(True)
  )
  cfg = RslRlOnPolicyRunnerCfg(num_steps_per_env=2, upload_model=False)
  cfg.actor.hidden_dims = (4,)
  cfg.critic.hidden_dims = (4,)
  cfg.algorithm.num_mini_batches = 1
  cfg.algorithm.num_learning_epochs = 1
  if configured:
    cfg.check_for_nan = False
    cfg.torch_compile_mode = "default"
  runner = MjlabOnPolicyRunner(TensorEnv(), asdict(cfg))
  runner.learn(1)
  assert compile_modes == (["default", "default"] if configured else [None, None])
  assert len(checks) == (0 if configured else 2)
  assert runner.alg.optimizer.state
