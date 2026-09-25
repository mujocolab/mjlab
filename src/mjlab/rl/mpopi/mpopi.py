"""MPOPI: turns off-policy replay segments into an importance-corrected batch.

MPOPI has no parameters and never updates the policy. At the start of each PPO
update it re-evaluates replay data under the current policy ``pi_old`` and
critic, and produces PPO-compatible samples whose surrogate weights are
``w = clip(pi_old / mu)`` and whose advantages/returns are V-trace estimates.
PPO's clipped ratio stays ``pi_theta / pi_old``; the weight is a constant.
"""

from dataclasses import dataclass, field

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.rl.mpopi.config import MpopiCfg
from mjlab.rl.mpopi.estimators import (
  bootstrap_time_outs,
  effective_sample_size,
  importance_log_ratio,
  importance_weights,
  vtrace,
)
from mjlab.rl.mpopi.replay_buffer import ReplayBuffer


@dataclass
class MpopiBatch:
  """Flat ``[n]`` replay samples prepared for PPO.

  Field names mirror ``rsl_rl.storage.RolloutStorage.Batch``. ``old_*`` fields
  refer to ``pi_old`` (the policy at the start of the update, recomputed), not
  to the behavior policy, which is kept separately in
  ``behavior_actions_log_prob``.
  """

  observations: TensorDict
  actions: torch.Tensor
  values: torch.Tensor
  """``V_{phi_k}(s)`` from the critic at the start of the update."""
  advantages: torch.Tensor
  """Truncated-IS GAE advantages (unnormalized)."""
  returns: torch.Tensor
  """V-trace value targets."""
  old_actions_log_prob: torch.Tensor
  old_distribution_params: tuple[torch.Tensor, ...]
  behavior_actions_log_prob: torch.Tensor
  weights: torch.Tensor
  """Surrogate weights; 0 for rejected samples, 1 for accepted in naive mode."""
  mask: torch.Tensor
  """Bool, True for accepted samples."""
  policy_age: torch.Tensor
  metrics: dict[str, float] = field(default_factory=dict)

  def __len__(self) -> int:
    return self.actions.shape[0]


class Mpopi:
  """Replay sample selection, distribution correction and PPO batch building."""

  def __init__(self, cfg: MpopiCfg, gamma: float, lam: float) -> None:
    cfg.validate()
    if not cfg.enabled:
      raise ValueError("Mpopi requires mode 'naive_replay_ppo' or 'mpopi_ppo'.")
    self.cfg = cfg
    self.gamma = gamma
    self.lam = lam

  @property
  def corrects(self) -> bool:
    """Whether importance correction is applied (False in naive mode)."""
    return self.cfg.mode == "mpopi_ppo"

  def replay_size(self, num_fresh: int) -> int:
    if self.cfg.replay_batch_size is not None:
      return self.cfg.replay_batch_size
    return int(round(self.cfg.replay_ratio * num_fresh))

  def process(
    self,
    buffer: ReplayBuffer,
    actor: MLPModel,
    critic: MLPModel,
    version: int,
    num_fresh: int,
  ) -> tuple[MpopiBatch | None, dict[str, float]]:
    """Build the replay part of PPO's batch.

    Args:
      buffer: Replay buffer of past segments (all with version < ``version``).
      actor: Current actor, i.e. ``pi_old`` for this update.
      critic: Current critic.
      version: Current iteration, used for policy age.
      num_fresh: Number of fresh on-policy samples (for ``replay_ratio``).

    Returns:
      The replay batch (None when nothing is available) and a metrics dict.
    """
    cfg = self.cfg
    metrics: dict[str, float] = {
      "buffer_segments": float(len(buffer)),
      "buffer_transitions": float(buffer.num_transitions),
    }
    n_request = self.replay_size(num_fresh)
    all_slots = buffer.slots(version)
    slots = buffer.slots(version, cfg.max_policy_age)
    num_stale = len(all_slots) - len(slots)
    metrics["stale_frac"] = num_stale / len(all_slots) if all_slots else 0.0
    if not slots or (n_request == 0 and cfg.sampling_strategy == "uniform"):
      return None, metrics
    assert buffer.observations is not None
    assert buffer.bootstrap_observations is not None

    with torch.no_grad():
      seg = self._evaluate_segments(buffer, slots, actor, critic, version)

    num_total = seg["actions"].shape[0]
    if cfg.sampling_strategy == "all":
      idx = torch.arange(num_total, device=buffer.device)
    else:
      perm = torch.randperm(num_total, device=buffer.device)
      idx = perm[: min(n_request, num_total)]
    s = {k: v[idx] for k, v in seg.items() if isinstance(v, torch.Tensor)}
    params = tuple(p[idx] for p in seg["old_distribution_params"])
    observations = seg["observations"][idx]

    # Rejection masks. They are computed from the true ratio in both modes so
    # that naive replay and MPOPI select exactly the same samples.
    log_ratio = importance_log_ratio(
      s["old_actions_log_prob"], s["behavior_actions_log_prob"]
    )
    finite = torch.isfinite(log_ratio)
    mask = finite.clone()
    rejected_log_ratio = torch.zeros_like(mask)
    if cfg.max_abs_log_ratio is not None:
      rejected_log_ratio = finite & (log_ratio.abs() > cfg.max_abs_log_ratio)
      mask &= ~rejected_log_ratio

    weights, clipped = importance_weights(
      log_ratio,
      clip_min=cfg.importance_weight_clip_min,
      clip_max=cfg.importance_weight_clip_max,
      log_ratio_clamp=cfg.log_ratio_clamp,
    )
    raw_ratio = torch.exp(
      torch.where(finite, log_ratio, torch.zeros_like(log_ratio)).clamp(
        -cfg.log_ratio_clamp, cfg.log_ratio_clamp
      )
    )
    ess = effective_sample_size(weights[mask])
    ess_gated = cfg.min_ess > 0.0 and bool(ess < cfg.min_ess)
    num_accepted_pre_gate = int(mask.sum())
    if ess_gated:
      mask = torch.zeros_like(mask)

    if self.corrects:
      weights = weights * mask
      if cfg.weight_normalization == "self_normalized" and mask.any():
        weights = weights / weights[mask].mean()
    else:
      weights = mask.to(weights.dtype)

    n = idx.numel()
    num_accepted = int(mask.sum())
    metrics.update(
      {
        "sampled": float(n),
        "accepted": float(num_accepted),
        "rejected": float(n - num_accepted),
        "rejected_nonfinite": float((~finite).sum()),
        "rejected_log_ratio": float(rejected_log_ratio.sum()),
        "rejected_ess": float(num_accepted_pre_gate if ess_gated else 0),
        "ess": float(ess),
        "policy_age_mean": float(s["policy_age"].float().mean()),
        "policy_age_max": float(s["policy_age"].max()),
      }
    )
    if num_accepted > 0:
      w = weights[mask]
      r = raw_ratio[mask]
      metrics.update(
        {
          "weight_mean": float(w.mean()),
          "weight_std": float(w.std(unbiased=False)),
          "weight_min": float(w.min()),
          "weight_max": float(w.max()),
          "raw_ratio_mean": float(r.mean()),
          "raw_ratio_std": float(r.std(unbiased=False)),
          "raw_ratio_min": float(r.min()),
          "raw_ratio_max": float(r.max()),
          "clipped_frac": float(clipped[mask].float().mean()),
          "behavior_kl": float(s["behavior_kl"][mask.squeeze(-1)].mean()),
        }
      )

    batch = MpopiBatch(
      observations=observations,
      actions=s["actions"],
      values=s["values"],
      advantages=s["advantages"],
      returns=s["returns"],
      old_actions_log_prob=s["old_actions_log_prob"],
      old_distribution_params=params,
      behavior_actions_log_prob=s["behavior_actions_log_prob"],
      weights=weights,
      mask=mask,
      policy_age=s["policy_age"],
      metrics=metrics,
    )
    return batch, metrics

  def _evaluate_segments(
    self,
    buffer: ReplayBuffer,
    slots: list[int],
    actor: MLPModel,
    critic: MLPModel,
    version: int,
  ) -> dict:
    """Recompute pi_old log-probs, values and V-trace targets for segments.

    Returns flat ``[S * T * N, ...]`` tensors (segment-major, then time, env).
    """
    assert buffer.observations is not None
    assert buffer.bootstrap_observations is not None
    num_steps, num_envs = buffer.observations.batch_size[1:3]
    logp, values, boot_values, params, kl = [], [], [], [], []
    for i in slots:
      obs = buffer.observations[i].flatten(0, 1)
      actor(obs, stochastic_output=True)
      logp.append(
        actor.get_output_log_prob(buffer.actions[i].flatten(0, 1)).view(
          num_steps, num_envs, 1
        )
      )
      p_old = tuple(
        p.view(num_steps, num_envs, *p.shape[1:])
        for p in actor.output_distribution_params
      )
      params.append(p_old)
      p_mu = tuple(p[i] for p in buffer.behavior_distribution_params)
      kl.append(actor.get_kl_divergence(p_mu, p_old))
      values.append(critic(obs).view(num_steps, num_envs, 1))
      boot_values.append(critic(buffer.bootstrap_observations[i]))

    slot_t = torch.tensor(slots, device=buffer.device)
    old_logp = torch.stack(logp)  # [S, T, N, 1]
    beh_logp = buffer.behavior_log_prob[slot_t]
    values_t = torch.stack(values)
    rewards = bootstrap_time_outs(
      buffer.rewards[slot_t], buffer.time_outs[slot_t], values_t, self.gamma
    )
    dones = buffer.dones[slot_t]
    if self.corrects:
      trace_log_ratio = importance_log_ratio(old_logp, beh_logp)
    else:
      trace_log_ratio = torch.zeros_like(old_logp)

    # Segments are independent, so fold them into the env axis: [T, S*N, 1].
    def fold(x: torch.Tensor) -> torch.Tensor:
      return x.transpose(0, 1).flatten(1, 2)

    def unfold(x: torch.Tensor) -> torch.Tensor:
      return x.view(num_steps, len(slots), num_envs, 1).transpose(0, 1)

    returns, advantages = vtrace(
      rewards=fold(rewards),
      dones=fold(dones),
      values=fold(values_t),
      bootstrap_values=torch.stack(boot_values).flatten(0, 1),
      log_ratio=fold(trace_log_ratio),
      gamma=self.gamma,
      lam=self.lam,
      rho_clip=self.cfg.importance_weight_clip_max,
      trace_clip=self.cfg.trace_clip_max,
      log_ratio_clamp=self.cfg.log_ratio_clamp,
    )
    versions = buffer.policy_version[slots].to(buffer.device)
    ages = (version - versions).view(-1, 1, 1, 1)

    def flat(x: torch.Tensor) -> torch.Tensor:
      return x.flatten(0, 2)

    num_params = len(params[0])
    return {
      "observations": buffer.observations[slot_t].flatten(0, 2),
      "actions": flat(buffer.actions[slot_t]),
      "values": flat(values_t),
      "returns": flat(unfold(returns)),
      "advantages": flat(unfold(advantages)),
      "old_actions_log_prob": flat(old_logp),
      "behavior_actions_log_prob": flat(beh_logp),
      "old_distribution_params": tuple(
        flat(torch.stack([p[j] for p in params])) for j in range(num_params)
      ),
      "behavior_kl": torch.stack(kl).flatten(0, 2),
      "policy_age": flat(ages.expand(-1, num_steps, num_envs, 1)),
    }
