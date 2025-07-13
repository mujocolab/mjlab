"""Useful methods for MPD terminations."""

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
  return env.episode_length_buf >= env.max_episode_length
