"""Test cartpole training and reward threshold."""

import tempfile
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import pytest
import torch
from rsl_rl.runners import OnPolicyRunner

from conftest import get_test_device
from mjlab.rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from mjlab.utils.torch import configure_torch_backends
from assets.cartpole.cartpole_env_cfg import create_cartpole_env_cfg


@pytest.mark.slow
def test_cartpole_upright_reward_threshold():
  """Test that cartpole training achieves upright reward > 3.00."""
  configure_torch_backends()
  device = get_test_device()
  if device == "cuda":
    device = f"cuda:{torch.cuda.current_device()}"

  # Create environment
  env_cfg = create_cartpole_env_cfg()
  env_cfg.scene.num_envs = 32
  env_cfg.episode_length_s = 10.0

  env = gym.make("Mjlab-Cartpole", cfg=env_cfg, device=device, render_mode=None)

  # Create runner config
  runner_cfg = RslRlOnPolicyRunnerCfg()
  runner_cfg.max_iterations = 40
  runner_cfg.experiment_name = "cartpole_test"
  runner_cfg.logger = "tensorboard"
  runner_cfg.wandb_project = "test"

  with tempfile.TemporaryDirectory() as tmpdir:
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(env_wrapped, asdict(runner_cfg), str(Path(tmpdir) / "logs"), device)

    # Train
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    # Evaluate: collect rewards from completed episodes
    policy = runner.get_inference_policy(device=device)
    upright_rewards = []
    obs = env_wrapped.get_observations()
    for _ in range(env.unwrapped.max_episode_length):
      actions = policy(obs)
      obs, _, dones, extras = env_wrapped.step(actions)

      # Collect upright reward when episodes finish
      if dones.any() and "log" in extras and "Episode_Reward/upright" in extras["log"]:
        upright_rewards.append(extras["log"]["Episode_Reward/upright"].item())

    env.close()

    # Check threshold
    assert upright_rewards, "No rewards collected"
    max_reward = max(upright_rewards)
    print(f"\nMax upright reward: {max_reward:.4f} (from {len(upright_rewards)} episodes)")
    reward_threshold = 3.00
    assert max_reward > reward_threshold, f"Max reward {max_reward:.2f} ≤ {reward_threshold}"

