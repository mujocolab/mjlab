"""Pure-torch point-mass environment for fast MPOPI validation on CPU."""

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class PointMassVecEnv(VecEnv):
  """1-D point mass that must be driven to the origin.

  ``x' = x + dt * clip(a, -1, 1)``; reward ``-x^2 - 0.01 a^2``. Episodes
  terminate when ``|x| > 3`` and time out after ``max_episode_length`` steps.
  Initial positions are uniform in ``[-2, 2]``. No MuJoCo dependency.
  """

  def __init__(
    self,
    num_envs: int = 64,
    max_episode_length: int = 50,
    dt: float = 0.1,
    device: str = "cpu",
    seed: int = 0,
  ) -> None:
    self.num_envs = num_envs
    self.num_actions = 1
    self.max_episode_length = max_episode_length
    self.device = device
    self.cfg = {"name": "PointMass"}
    self.dt = dt
    self.generator = torch.Generator(device=device).manual_seed(seed)
    self.pos = torch.zeros(num_envs, 1, device=device)
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._reset(torch.ones(num_envs, dtype=torch.bool, device=device))

  def get_observations(self) -> TensorDict:
    return TensorDict(
      {"actor": self.pos.clone(), "critic": self.pos.clone()},
      batch_size=[self.num_envs],
    )

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    a = actions.clamp(-1.0, 1.0)
    self.pos = self.pos + self.dt * a
    reward = -(self.pos.square() + 0.01 * actions.square()).sum(-1)
    self.episode_length_buf += 1
    terminated = self.pos.abs().squeeze(-1) > 3.0
    time_outs = (self.episode_length_buf >= self.max_episode_length) & ~terminated
    dones = terminated | time_outs
    self._reset(dones)
    extras = {"time_outs": time_outs, "log": {}}
    return self.get_observations(), reward, dones.long(), extras

  def _reset(self, env_mask: torch.Tensor) -> None:
    n = int(env_mask.sum())
    if n == 0:
      return
    new = torch.rand(n, 1, generator=self.generator, device=self.device) * 4 - 2
    self.pos[env_mask] = new
    self.episode_length_buf[env_mask] = 0
