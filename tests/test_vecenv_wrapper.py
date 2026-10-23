"""RSL-RL episode-boundary and value-bootstrap behavior."""

from dataclasses import asdict

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  MjlabOnPolicyRunner,
  RslRlOnPolicyRunnerCfg,
  RslRlVecEnvWrapper,
)
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg


def _terminal_condition(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.arange(env.num_envs, device=env.device) % 2 == 0


@pytest.fixture
def wrapped_env(request):
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 4
  cfg.episode_length_s = 1.0
  cfg.is_finite_horizon = request.param
  cfg.terminations["terminal_condition"] = TerminationTermCfg(func=_terminal_condition)
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  wrapped = RslRlVecEnvWrapper(env)
  # Envs 1 and 2 reach the deadline; envs 0 and 2 also reach a terminal state.
  env.episode_length_buf[1:3] = env.max_episode_length - 1
  yield wrapped
  wrapped.close()


@pytest.mark.parametrize("wrapped_env", [False, True], indirect=True)
def test_wrapper_distinguishes_terminal_and_timeout(wrapped_env):
  obs, _, dones, extras = wrapped_env.step(
    torch.zeros(
      wrapped_env.num_envs, wrapped_env.num_actions, device=wrapped_env.device
    )
  )
  assert obs.batch_size == (4,)
  torch.testing.assert_close(
    dones, torch.tensor([1, 1, 1, 0], device=wrapped_env.device)
  )
  # Preserve both raw Gymnasium signals, including the simultaneous boundary.
  torch.testing.assert_close(
    wrapped_env.unwrapped.reset_terminated,
    torch.tensor([True, False, True, False], device=wrapped_env.device),
  )
  torch.testing.assert_close(
    wrapped_env.unwrapped.reset_time_outs,
    torch.tensor([False, True, True, False], device=wrapped_env.device),
  )
  if wrapped_env.cfg.is_finite_horizon:
    assert "time_outs" not in extras
  else:
    torch.testing.assert_close(
      extras["time_outs"],
      torch.tensor([False, True, False, False], device=wrapped_env.device),
    )


@pytest.mark.parametrize("wrapped_env", [False, True], indirect=True)
def test_ppo_bootstraps_only_nonterminal_timeouts(wrapped_env):
  agent_cfg = RslRlOnPolicyRunnerCfg(num_steps_per_env=1, upload_model=False)
  agent_cfg.actor.hidden_dims = (8,)
  agent_cfg.critic.hidden_dims = (8,)
  agent_cfg.algorithm.gamma = 0.9
  runner = MjlabOnPolicyRunner(
    wrapped_env, asdict(agent_cfg), device=str(wrapped_env.device)
  )
  # A constant V(s)=2 makes an incorrect terminal bootstrap visible as +1.8.
  with torch.no_grad():
    for parameter in runner.alg.critic.parameters():
      parameter.zero_()
    output = runner.alg.critic.mlp[-1]
    assert isinstance(output, torch.nn.Linear) and output.bias is not None
    output.bias.fill_(2.0)

  with torch.inference_mode():
    actions = runner.alg.act(wrapped_env.get_observations())
    obs, rewards, dones, extras = wrapped_env.step(actions)
    runner.alg.process_env_step(obs, rewards, dones, extras)

  expected = rewards.clone()
  if not wrapped_env.cfg.is_finite_horizon:
    expected[1] += 1.8
  torch.testing.assert_close(runner.alg.storage.rewards[0, :, 0], expected)
