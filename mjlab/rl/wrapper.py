import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from mjlab.envs import ManagerBasedRLEnv


class RslRlVecEnvWrapper(VecEnv):
  def __init__(self, env: ManagerBasedRLEnv):
    self.env = env

    self.num_envs = self.unwrapped.num_envs
    self.device = self.unwrapped.device
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim

    # reset at the start since the RSL-RL runner does not call reset
    self.env.reset()

  def __str__(self):
    """Returns the wrapper name and the :attr:`env` representation string."""
    return f"<{type(self).__name__}{self.env}>"

  def __repr__(self):
    """Returns the string representation of the wrapper."""
    return str(self)

  """
    Properties -- Gym.Wrapper
    """

  @property
  def cfg(self) -> object:
    """Returns the configuration class instance of the environment."""
    return self.unwrapped.cfg

  @classmethod
  def class_name(cls) -> str:
    """Returns the class name of the wrapper."""
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRLEnv:
    """Returns the base environment of the wrapper.

    This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
    """
    return self.env

  """
    Properties
    """

  @property
  def episode_length_buf(self) -> torch.Tensor:
    """The episode length buffer."""
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor):
    """Set the episode length buffer.

    Note:
        This is needed to perform random initialization of episode lengths in RSL-RL.
    """
    self.unwrapped.episode_length_buf = value

  """
    Operations - MDP
    """

  def seed(self, seed: int = -1) -> int:  # noqa: D102
    return self.unwrapped.seed(seed)

  def get_observations(self) -> TensorDict:
    """Returns the current observations of the environment."""
    obs_dict = self.unwrapped.observation_manager.compute()
    return TensorDict(obs_dict, batch_size=[self.num_envs])

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # record step information
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    # compute dones for compatibility with RSL-RL
    dones = (terminated | truncated).to(dtype=torch.long)
    # move time out information to the extras dict
    # this is only needed for infinite horizon tasks
    if not self.unwrapped.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    # return the step information
    return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

  def close(self):  # noqa: D102
    return self.env.close()
