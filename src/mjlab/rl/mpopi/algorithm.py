"""PPO with an MPOPI replay-correction stage in front of the update.

``MpopiPpo`` is selected through RSL-RL's ``algorithm.class_name`` and is built
by the unmodified ``PPO.construct_algorithm``. It keeps PPO as the only
optimizer. MPOPI only prepares data: at the start of ``update()`` it turns the
replay buffer into extra weighted samples, which are mixed with the fresh
rollout and optimized with PPO's clipped objective.

``update()`` mirrors ``rsl_rl.algorithms.PPO.update`` from rsl-rl-lib 5.5.1
because the upstream loss is inline and has no hook for per-sample weights.
When no replay sample is available it takes exactly the upstream code path, so
the resulting parameters are identical to plain PPO.

In mode ``"mpc_ppo"`` the replay source is MPC-generated data instead of past
PPO rollouts: an attached :class:`mjlab.mpc.collector.MpcCollector` fills a
separate buffer, the same MPOPI estimators correct it, and an annealed
behavior-cloning term pulls the policy mean toward the MPC action.
"""

from collections.abc import Callable
from dataclasses import asdict, fields, replace
from typing import Any, Generator, Protocol, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.rl.mpopi.config import MpopiCfg
from mjlab.rl.mpopi.mpopi import Mpopi, MpopiBatch
from mjlab.rl.mpopi.replay_buffer import ReplayBuffer

_Pool = dict[str, Any]


class MpcDataSource(Protocol):
  """What ``MpopiPpo`` needs from an MPC collector."""

  def collect(
    self, policy: Callable[[TensorDict], torch.Tensor] | None = None
  ) -> tuple[dict, dict[str, float]]: ...

  def close(self) -> None: ...


class MpopiPpo(PPO):
  """PPO whose update batch is fresh data plus MPOPI-corrected replay data."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    mpopi: dict[str, Any] | MpopiCfg | None = None,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    if isinstance(mpopi, dict):
      mpopi = MpopiCfg.from_dict(cast(dict[str, Any], mpopi))
    cfg = mpopi if mpopi is not None else MpopiCfg(mode="mpopi_ppo")
    if actor.is_recurrent or critic.is_recurrent:
      raise ValueError("MPOPI does not support recurrent actors or critics.")
    if self.rnd is not None or self.symmetry is not None:
      raise ValueError("MPOPI does not support RND or symmetry augmentation.")
    self.mpopi_cfg = cfg
    self.policy_version = 0
    self.mpc_cfg = cfg.mpc if cfg.mode == "mpc_ppo" else None
    self.mpc_collector: MpcDataSource | None = None
    if self.mpc_cfg is None:
      self.mpopi = Mpopi(cfg, gamma=self.gamma, lam=self.lam)
      self.replay = ReplayBuffer(cfg.replay_buffer_size, device=self.device)
    else:
      cfg.validate()
      if not isinstance(actor.distribution, GaussianDistribution):
        raise ValueError("mpc_ppo requires a Gaussian actor distribution.")
      # MPC data reuses MPOPI's estimators; every segment in the buffer is used.
      mode = "mpopi_ppo" if self.mpc_cfg.correction else "naive_replay_ppo"
      fraction = self.mpc_cfg.inject_fraction
      if fraction is None:
        mpc_mpopi_cfg = replace(
          cfg, mode=mode, sampling_strategy="all", max_policy_age=None
        )
      else:
        # A fraction p of the batch is MPC: p / (1 - p) MPC samples per fresh one.
        mpc_mpopi_cfg = replace(
          cfg,
          mode=mode,
          sampling_strategy="uniform",
          replay_ratio=fraction / (1.0 - fraction),
          replay_batch_size=None,
          max_policy_age=None,
        )
      self.mpopi = Mpopi(mpc_mpopi_cfg, gamma=self.gamma, lam=self.lam)
      self.replay = ReplayBuffer(self.mpc_cfg.buffer_segments, device=self.device)
    # Replay of PPO's own past rollouts: the replay modes, or mpc_ppo with
    # ``replay_own_rollouts`` (then next to the MPC buffer).
    self.own_mpopi: Mpopi | None = None
    self.own_replay: ReplayBuffer | None = None
    if self.mpc_cfg is None:
      self.own_mpopi, self.own_replay = self.mpopi, self.replay
    elif self.mpc_cfg.replay_own_rollouts:
      own_cfg = replace(cfg, mode="mpopi_ppo")
      self.own_mpopi = Mpopi(own_cfg, gamma=self.gamma, lam=self.lam)
      self.own_replay = ReplayBuffer(cfg.replay_buffer_size, device=self.device)
    shape = (storage.num_transitions_per_env, storage.num_envs, 1)
    self._raw_rewards = torch.zeros(shape, device=self.device)
    self._time_outs = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._bootstrap_obs: TensorDict | None = None

  def attach_mpc_collector(self, collector: MpcDataSource) -> None:
    """Set the MPC data source (mode ``"mpc_ppo"`` only)."""
    if self.mpc_cfg is None:
      raise ValueError("An MPC collector requires mode 'mpc_ppo'.")
    self.mpc_collector = collector

  # Rollout hooks.

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    # Record raw rewards and time-outs before PPO folds the stale time-out
    # bootstrap gamma * V_old(s_t) into the stored rewards.
    step = self.storage.step
    self._raw_rewards[step].copy_(rewards.view(-1, 1))
    if "time_outs" in extras:
      self._time_outs[step].copy_(extras["time_outs"].view(-1, 1))
    else:
      self._time_outs[step].zero_()
    super().process_env_step(obs, rewards, dones, extras)

  def compute_returns(self, obs: TensorDict) -> None:
    super().compute_returns(obs)
    self._bootstrap_obs = obs.clone()

  # Update.

  def update(self) -> dict[str, float]:
    st = self.storage
    num_fresh = st.num_envs * st.num_transitions_per_env
    bc_weight = 0.0
    mpc_metrics: dict[str, float] = {}
    if self.mpc_cfg is not None:
      mpc_metrics = self._refresh_mpc_buffer()
      bc_weight = self.mpc_cfg.bc_weight(self.policy_version)
      use_in_ppo = self.mpc_cfg.use_in_ppo
    else:
      use_in_ppo = True
    if use_in_ppo:
      replay, mpopi_metrics = self.mpopi.process(
        self.replay, self.actor, self.critic, self.policy_version, num_fresh
      )
    elif bc_weight > 0.0:
      # Behavior cloning only: no importance weights or V-trace are needed,
      # and the data may have no behavior density (execution_std 0, DAgger).
      replay, mpopi_metrics = self._bc_batch(), {}
    else:
      replay, mpopi_metrics = None, {}
    if replay is not None and not use_in_ppo:
      # Behavior cloning only: MPC samples stay out of PPO's losses.
      replay.mask = torch.zeros_like(replay.mask)
      replay.weights = torch.zeros_like(replay.weights)
    if replay is not None and not bool(replay.mask.any()) and bc_weight == 0.0:
      replay = None  # Everything rejected: fall back to plain PPO.
    num_own = 0
    if self.mpc_cfg is not None and self.own_mpopi is not None:
      assert self.own_replay is not None
      own, mpopi_metrics = self.own_mpopi.process(
        self.own_replay, self.actor, self.critic, self.policy_version, num_fresh
      )
      if own is not None and bool(own.mask.any()):
        # Own replay first, then the behavior-cloning-only MPC samples.
        num_own = own.actions.shape[0]
        replay = own if replay is None else _concat_batches(own, replay)
    pool = self._build_pool(replay, num_own)
    weighted = replay is not None
    mean_bc_loss = 0.0

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_kl = 0.0
    mean_clip_fraction = 0.0

    for batch, weights, mask, bc in self._mini_batch_generator(pool):
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = _normalize(batch.advantages, mask)  # type: ignore[arg-type]

      with torch.amp.autocast(  # pyright: ignore[reportPrivateImportUsage]
        device_type=torch.device(self.device).type,
        enabled=self.use_mixed_precision,
        dtype=torch.bfloat16,
      ):
        self.actor(batch.observations, stochastic_output=True)
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
        values = self.critic(batch.observations)
        distribution_params = self.actor.output_distribution_params
        entropy = self.actor.output_entropy

        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params,  # type: ignore[arg-type]
            distribution_params,
          )
          kl_mean = _mean(kl, mask)
          if self.desired_kl is not None and self.schedule == "adaptive":
            self._adapt_learning_rate(kl_mean)

        assert batch.old_actions_log_prob is not None
        assert batch.advantages is not None
        assert batch.values is not None and batch.returns is not None
        surrogate_loss, ratio = weighted_clipped_surrogate(
          actions_log_prob,
          torch.squeeze(batch.old_actions_log_prob),
          torch.squeeze(batch.advantages),
          self.clip_param,
          weights=torch.squeeze(weights, -1) if weighted else None,  # type: ignore[arg-type]
          mask=mask,
        )

        if self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(
            -self.clip_param, self.clip_param
          )
          value_losses = (values - batch.returns).pow(2)
          value_losses_clipped = (value_clipped - batch.returns).pow(2)
          value_loss = _mean(torch.max(value_losses, value_losses_clipped), mask)
        else:
          value_loss = _mean((batch.returns - values).pow(2), mask)

        entropy_mean = _mean(entropy, mask)
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy_mean
        )
        if bc_weight > 0.0 and bc is not None:
          bc_target, bc_mask = bc
          bc_error = (distribution_params[0] - bc_target).square().mean(dim=-1)
          bc_loss = _mean(bc_error, bc_mask)
          loss = loss + bc_weight * bc_loss
          mean_bc_loss += bc_loss.item()

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      with torch.no_grad():
        clipped = ((ratio - 1.0).abs() > self.clip_param).float()
        mean_clip_fraction += _mean(clipped, mask).item()
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_mean.item()
      mean_kl += kl_mean.item()

    # Normalizers are updated on fresh on-policy observations only (upstream).
    obs = cast(TensorDict, st.observations.flatten(0, 1))
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "kl": mean_kl / num_updates,
      "clip_fraction": mean_clip_fraction / num_updates,
    }
    num_accepted = int(replay.mask.sum()) if replay is not None else 0
    mpopi_metrics["gradient_samples"] = float(num_fresh + num_accepted)
    loss_dict.update({f"mpopi/{k}": v for k, v in mpopi_metrics.items()})
    if self.mpc_cfg is not None:
      mpc_metrics["bc_weight"] = bc_weight
      mpc_metrics["bc_loss"] = mean_bc_loss / num_updates
      loss_dict.update({f"mpc/{k}": v for k, v in mpc_metrics.items()})
    if self.own_replay is not None:
      self._store_fresh_segment(self.own_replay)
    self.policy_version += 1
    st.clear()
    return loss_dict

  def _refresh_mpc_buffer(self) -> dict[str, float]:
    """Collect a new MPC segment when scheduled and drop stale ones."""
    cfg, version = self.mpc_cfg, self.policy_version
    assert cfg is not None
    metrics: dict[str, float] = {}
    if cfg.collects(version):
      if self.mpc_collector is None:
        raise RuntimeError("mpc_ppo needs attach_mpc_collector() before training.")
      policy = self._act_stochastic if cfg.driver == "policy" else None
      segment, collect_metrics = self.mpc_collector.collect(policy)
      self.replay.insert(**segment, policy_version=version)
      metrics.update({f"collect_{k}": v for k, v in collect_metrics.items()})
    if cfg.max_age is not None:
      self.replay.evict_older_than(version, cfg.max_age)
    metrics["buffer_segments"] = float(len(self.replay))
    return metrics

  def _act_stochastic(self, obs: TensorDict) -> torch.Tensor:
    """Sample from the current policy (DAgger's acting policy)."""
    with torch.no_grad():
      return self.actor(obs, stochastic_output=True)

  def _bc_batch(self) -> MpopiBatch | None:
    """Every MPC sample in the buffer as a behavior-cloning-only batch.

    The PPO fields are filled with the current actor and critic so that the
    (masked-out) PPO terms stay finite; only the behavior mean ``u0`` is used.
    """
    slots = self.replay.slots(self.policy_version)
    if not slots:
      return None
    buffer = self.replay
    assert buffer.observations is not None
    idx = torch.tensor(slots, device=buffer.device)
    obs = cast(TensorDict, buffer.observations[idx].flatten(0, 2))
    actions = buffer.actions[idx].flatten(0, 2)
    behavior = tuple(p[idx].flatten(0, 2) for p in buffer.behavior_distribution_params)
    with torch.no_grad():
      self.actor(obs, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(actions).view(-1, 1)
      params = tuple(p.clone() for p in self.actor.output_distribution_params)
      values = self.critic(obs)
    zeros = torch.zeros_like(values)
    return MpopiBatch(
      observations=obs,
      actions=actions,
      values=values,
      advantages=zeros,
      returns=values,
      old_actions_log_prob=log_prob,
      old_distribution_params=params,
      behavior_actions_log_prob=zeros,
      behavior_distribution_params=behavior,
      weights=zeros,
      mask=torch.zeros_like(values, dtype=torch.bool),
      policy_age=torch.zeros_like(values, dtype=torch.long),
    )

  def save(self) -> dict:
    saved = super().save()
    saved["mpopi_cfg"] = asdict(self.mpopi_cfg)
    return saved

  # Private helpers.

  def _adapt_learning_rate(self, kl_mean: torch.Tensor) -> None:
    """Upstream adaptive KL learning-rate rule (``ppo.py:246-266``)."""
    if self.is_multi_gpu:
      dist.all_reduce(kl_mean, op=dist.ReduceOp.SUM)  # ty: ignore[possibly-missing-attribute]
      kl_mean /= self.gpu_world_size
    if self.gpu_global_rank == 0:
      if kl_mean > self.desired_kl * 2.0:
        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
      elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
    if self.is_multi_gpu:
      lr_tensor = torch.tensor(self.learning_rate, device=self.device)
      dist.broadcast(lr_tensor, src=0)  # ty: ignore[possibly-missing-attribute]
      self.learning_rate = lr_tensor.item()
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = self.learning_rate

  def _build_pool(self, replay: MpopiBatch | None, num_own: int = 0) -> _Pool:
    """Flatten the fresh rollout and append the replay batch, if any.

    The first ``num_own`` replay samples are PPO's own past data, which get no
    behavior-cloning target.
    """
    st = self.storage
    assert st.distribution_params is not None
    fresh: _Pool = {
      "observations": st.observations.flatten(0, 1),
      "actions": st.actions.flatten(0, 1),
      "values": st.values.flatten(0, 1),
      "returns": st.returns.flatten(0, 1),
      "advantages": st.advantages.flatten(0, 1),
      "old_actions_log_prob": st.actions_log_prob.flatten(0, 1),
      "old_distribution_params": tuple(p.flatten(0, 1) for p in st.distribution_params),
      "weights": None,
      "mask": None,
      "bc": None,
    }
    if replay is None:
      return fresh

    # Upstream already normalized the fresh advantages on their own; recover
    # the raw ones so fresh and replay are normalized together.
    fresh_adv = (st.returns - st.values).flatten(0, 1)
    advantages = torch.cat([fresh_adv, replay.advantages])
    num_fresh = fresh_adv.shape[0]
    ones = torch.ones(num_fresh, 1, device=self.device)
    mask = torch.cat([ones.bool(), replay.mask])
    advantages = torch.where(mask, advantages, torch.zeros_like(advantages))
    if not self.normalize_advantage_per_mini_batch:
      advantages = _normalize(advantages, mask)
    return {
      "observations": TensorDict.cat([fresh["observations"], replay.observations]),
      "actions": torch.cat([fresh["actions"], replay.actions]),
      "values": torch.cat([fresh["values"], replay.values]),
      "returns": torch.cat([fresh["returns"], replay.returns]),
      "advantages": advantages,
      "old_actions_log_prob": torch.cat(
        [fresh["old_actions_log_prob"], replay.old_actions_log_prob]
      ),
      "old_distribution_params": tuple(
        torch.cat([f, r])
        for f, r in zip(
          fresh["old_distribution_params"],
          replay.old_distribution_params,
          strict=True,
        )
      ),
      "weights": torch.cat([ones, replay.weights]),
      "mask": mask,
      "bc": self._bc_targets(replay, num_fresh, num_own),
    }

  def _bc_targets(
    self, replay: MpopiBatch, num_fresh: int, num_own: int
  ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Behavior-cloning targets (MPC mean action) and mask over the pool.

    Fresh samples and the first ``num_own`` replay samples (PPO's own past
    data) are not MPC samples and are masked out.
    """
    if self.mpc_cfg is None:
      return None
    behavior_mean = replay.behavior_distribution_params[0]
    target = torch.cat(
      [behavior_mean.new_zeros(num_fresh, *behavior_mean.shape[1:]), behavior_mean]
    )
    is_mpc = torch.zeros(target.shape[0], 1, dtype=torch.bool, device=self.device)
    is_mpc[num_fresh + num_own :] = True
    return target, is_mpc

  def _mini_batch_generator(
    self, pool: _Pool
  ) -> Generator[
    tuple[
      RolloutStorage.Batch,
      torch.Tensor | None,
      torch.Tensor | None,
      tuple[torch.Tensor, torch.Tensor] | None,
    ],
    None,
    None,
  ]:
    """Same shuffling as ``RolloutStorage.mini_batch_generator``, over the pool."""
    batch_size = pool["actions"].shape[0]
    mini_batch_size = batch_size // self.num_mini_batches
    indices = torch.randperm(
      self.num_mini_batches * mini_batch_size,
      requires_grad=False,
      device=self.device,
    )
    for _ in range(self.num_learning_epochs):
      for i in range(self.num_mini_batches):
        idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
        batch = RolloutStorage.Batch(
          observations=pool["observations"][idx],
          actions=pool["actions"][idx],
          values=pool["values"][idx],
          advantages=pool["advantages"][idx],
          returns=pool["returns"][idx],
          old_actions_log_prob=pool["old_actions_log_prob"][idx],
          old_distribution_params=tuple(
            p[idx] for p in pool["old_distribution_params"]
          ),
        )
        weights = pool["weights"][idx] if pool["weights"] is not None else None
        mask = pool["mask"][idx] if pool["mask"] is not None else None
        bc = tuple(x[idx] for x in pool["bc"]) if pool["bc"] is not None else None
        yield batch, weights, mask, bc

  def _store_fresh_segment(self, buffer: ReplayBuffer) -> None:
    st = self.storage
    assert self._bootstrap_obs is not None, "compute_returns() must run first."
    assert st.distribution_params is not None
    buffer.insert(
      observations=st.observations,
      actions=st.actions,
      rewards=self._raw_rewards,
      dones=st.dones,
      time_outs=self._time_outs,
      behavior_log_prob=st.actions_log_prob,
      behavior_distribution_params=st.distribution_params,
      bootstrap_observations=self._bootstrap_obs,
      policy_version=self.policy_version,
    )


def _concat_batches(a: MpopiBatch, b: MpopiBatch) -> MpopiBatch:
  """Concatenate two replay batches along the sample dimension."""
  merged = {}
  for f in fields(a):
    x, y = getattr(a, f.name), getattr(b, f.name)
    if isinstance(x, tuple):
      merged[f.name] = tuple(torch.cat([p, q]) for p, q in zip(x, y, strict=True))
    elif isinstance(x, TensorDict):
      merged[f.name] = TensorDict.cat([x, y])
    elif isinstance(x, dict):
      merged[f.name] = {**y, **x}  # Metrics: keep the own-replay values.
    else:
      merged[f.name] = torch.cat([x, y])
  return MpopiBatch(**merged)


def weighted_clipped_surrogate(
  log_prob: torch.Tensor,
  old_log_prob: torch.Tensor,
  advantages: torch.Tensor,
  clip_param: float,
  weights: torch.Tensor | None = None,
  mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """MPOPI's decoupled clipped surrogate loss (to minimize).

  ``mean_i[ -w_i * min(r_i A_i, clip(r_i, 1 - eps, 1 + eps) A_i) ]`` over
  accepted samples, with ``r = pi_theta / pi_old`` and constant weights
  ``w = clip(pi_old / mu)``. With ``weights`` and ``mask`` both None this is
  exactly RSL-RL's PPO surrogate.

  Returns:
    The loss and the ratio ``r``.
  """
  ratio = torch.exp(log_prob - old_log_prob)
  surrogate = -advantages * ratio
  surrogate_clipped = -advantages * torch.clamp(
    ratio, 1.0 - clip_param, 1.0 + clip_param
  )
  per_sample = torch.max(surrogate, surrogate_clipped)
  if weights is not None:
    per_sample = per_sample * weights
  return _mean(per_sample, mask), ratio


def _mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
  """Mean over accepted samples; plain ``mean()`` when there is no mask."""
  if mask is None:
    return x.mean()
  m = mask.reshape(x.shape).to(x.dtype)
  return (x * m).sum() / m.sum().clamp_min(1.0)


def _normalize(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
  """Upstream advantage normalization, restricted to accepted samples."""
  if mask is None:
    return (x - x.mean()) / (x.std() + 1e-8)
  accepted = x[mask]
  out = (x - accepted.mean()) / (accepted.std() + 1e-8)
  return torch.where(mask, out, torch.zeros_like(out))
