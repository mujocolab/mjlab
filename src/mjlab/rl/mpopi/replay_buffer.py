"""GPU ring buffer of past rollout segments for MPOPI."""

import torch
from tensordict import TensorDict


class ReplayBuffer:
  """FIFO buffer of full ``[T, N]`` rollout segments.

  Segments are kept contiguous (rather than as independent transitions) so that
  V-trace targets can be recomputed with the current critic at every update.
  Storage is allocated lazily on the first insert, on ``device``, and reused;
  inserting copies device-to-device.

  Per-step fields (shape ``[K, T, N, ...]``): observations, actions, raw rewards
  (before RSL-RL's time-out bootstrap), dones, time_outs, behavior log-probs and
  behavior distribution params. Per-segment fields: bootstrap observations
  ``[K, N]`` (state after the last step) and the policy version ``[K]``.
  """

  def __init__(self, capacity: int, device: str | torch.device = "cpu") -> None:
    if capacity < 1:
      raise ValueError("capacity must be >= 1.")
    self.capacity = capacity
    self.device = torch.device(device)
    self._next = 0
    self.occupied = torch.zeros(capacity, dtype=torch.bool)
    self.policy_version = torch.full((capacity,), -1, dtype=torch.long)
    self.observations: TensorDict | None = None
    self.bootstrap_observations: TensorDict | None = None
    self.actions: torch.Tensor
    self.rewards: torch.Tensor
    self.dones: torch.Tensor
    self.time_outs: torch.Tensor
    self.behavior_log_prob: torch.Tensor
    self.behavior_distribution_params: tuple[torch.Tensor, ...]

  def __len__(self) -> int:
    return int(self.occupied.sum())

  @property
  def num_transitions(self) -> int:
    if self.observations is None:
      return 0
    num_steps, num_envs = self.observations.batch_size[1:3]
    return len(self) * num_steps * num_envs

  def insert(
    self,
    observations: TensorDict,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    time_outs: torch.Tensor,
    behavior_log_prob: torch.Tensor,
    behavior_distribution_params: tuple[torch.Tensor, ...],
    bootstrap_observations: TensorDict,
    policy_version: int,
  ) -> None:
    """Insert one segment, evicting the oldest when full.

    Args:
      observations: ``[T, N]`` TensorDict.
      actions: ``[T, N, A]``.
      rewards: ``[T, N, 1]`` raw rewards.
      dones: ``[T, N, 1]``.
      time_outs: ``[T, N, 1]``.
      behavior_log_prob: ``[T, N, 1]`` ``log mu(a|s)``.
      behavior_distribution_params: tuple of ``[T, N, ...]``.
      bootstrap_observations: ``[N]`` TensorDict.
      policy_version: Iteration whose policy collected the segment.
    """
    if self.observations is None:
      self._allocate(
        observations,
        actions,
        behavior_distribution_params,
        bootstrap_observations,
      )
    assert self.observations is not None
    assert self.bootstrap_observations is not None
    i = self._next
    self.observations[i].copy_(observations)
    self.bootstrap_observations[i].copy_(bootstrap_observations)
    self.actions[i].copy_(actions)
    self.rewards[i].copy_(rewards.view_as(self.rewards[i]))
    self.dones[i].copy_(dones.view_as(self.dones[i]))
    self.time_outs[i].copy_(time_outs.view_as(self.time_outs[i]))
    self.behavior_log_prob[i].copy_(
      behavior_log_prob.view_as(self.behavior_log_prob[i])
    )
    for dst, src in zip(
      self.behavior_distribution_params, behavior_distribution_params, strict=True
    ):
      dst[i].copy_(src)
    self.occupied[i] = True
    self.policy_version[i] = policy_version
    self._next = (i + 1) % self.capacity

  def slots(self, current_version: int, max_age: int | None = None) -> list[int]:
    """Occupied slot indices, oldest first, with ``age <= max_age``."""
    out = []
    for i in range(self.capacity):
      if not self.occupied[i]:
        continue
      age = current_version - int(self.policy_version[i])
      if max_age is None or age <= max_age:
        out.append(i)
    return sorted(out, key=lambda i: int(self.policy_version[i]))

  def clear(self) -> None:
    self.occupied.zero_()
    self.policy_version.fill_(-1)
    self._next = 0

  def _allocate(
    self,
    observations: TensorDict,
    actions: torch.Tensor,
    behavior_distribution_params: tuple[torch.Tensor, ...],
    bootstrap_observations: TensorDict,
  ) -> None:
    k, dev = self.capacity, self.device
    num_steps, num_envs = observations.batch_size[:2]

    def _zeros_like(t: torch.Tensor) -> torch.Tensor:
      return torch.zeros(k, *t.shape, dtype=t.dtype, device=dev)

    self.observations = TensorDict(
      {key: _zeros_like(v) for key, v in observations.items()},
      batch_size=[k, num_steps, num_envs],
      device=dev,
    )
    self.bootstrap_observations = TensorDict(
      {key: _zeros_like(v) for key, v in bootstrap_observations.items()},
      batch_size=[k, num_envs],
      device=dev,
    )
    self.actions = _zeros_like(actions)
    scalar_shape = (k, num_steps, num_envs, 1)
    self.rewards = torch.zeros(scalar_shape, device=dev)
    self.dones = torch.zeros(scalar_shape, dtype=torch.bool, device=dev)
    self.time_outs = torch.zeros(scalar_shape, dtype=torch.bool, device=dev)
    self.behavior_log_prob = torch.zeros(scalar_shape, device=dev)
    self.behavior_distribution_params = tuple(
      _zeros_like(p) for p in behavior_distribution_params
    )
