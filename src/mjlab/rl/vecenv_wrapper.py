from typing import Any, cast

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.spaces import Space


class RslRlVecEnvWrapper(VecEnv):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
  ):
    self.env = env
    self.clip_actions = clip_actions

    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    self._modify_action_space()

    # Reset at the start since rsl_rl does not call reset.
    self.env.reset()

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.unwrapped.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env

  # Properties.

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:  # type: ignore
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def get_observations(self) -> TensorDict:
    obs_dict = self.unwrapped.observation_manager.compute()
    return TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs])

  def reset(self) -> tuple[TensorDict, dict]:
    obs_dict, extras = self.env.reset()
    return TensorDict(
      cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]
    ), extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    term_or_trunc = terminated | truncated
    assert isinstance(rew, torch.Tensor)
    assert isinstance(term_or_trunc, torch.Tensor)
    dones = term_or_trunc.to(dtype=torch.long)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return (
      TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]),
      rew,
      dones,
      extras,
    )

  def close(self) -> None:
    return self.env.close()

  # Private methods.

  def _modify_action_space(self) -> None:
    if self.clip_actions is None:
      return

    from mjlab.spaces import Box, batch_space

    self.unwrapped.single_action_space = Box(
      shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
    )
    self.unwrapped.action_space = batch_space(
      self.unwrapped.single_action_space, self.num_envs
    )
