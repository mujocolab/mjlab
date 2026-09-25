"""Importance-sampling and V-trace estimators used by MPOPI.

All functions are pure tensor functions. Per-sample tensors follow the
RSL-RL storage convention of a trailing singleton dimension, e.g. ``[T, N, 1]``.
"""

import torch


def importance_log_ratio(
  target_log_prob: torch.Tensor, behavior_log_prob: torch.Tensor
) -> torch.Tensor:
  """``log(pi(a|s) / mu(a|s))`` computed in the log domain."""
  return target_log_prob - behavior_log_prob


def importance_weights(
  log_ratio: torch.Tensor,
  clip_min: float = 0.0,
  clip_max: float | None = None,
  log_ratio_clamp: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Convert log ratios into clipped importance weights.

  Non-finite log ratios map to weight 0 and should be masked by the caller.

  Returns:
    The weights and a bool tensor marking where clipping changed the weight.
  """
  finite = torch.isfinite(log_ratio)
  safe = torch.where(finite, log_ratio, torch.zeros_like(log_ratio))
  ratio = torch.exp(safe.clamp(-log_ratio_clamp, log_ratio_clamp))
  weights = ratio.clamp(min=clip_min, max=clip_max)
  clipped = (weights != ratio) & finite
  weights = torch.where(finite, weights, torch.zeros_like(weights))
  return weights, clipped


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
  """Normalized effective sample size ``(sum w)^2 / (n * sum w^2)`` in ``[0, 1]``.

  Returns 0 for an empty or all-zero weight vector.
  """
  n = weights.numel()
  sum_w = weights.sum()
  sum_w2 = weights.square().sum()
  if n == 0:
    return torch.zeros((), device=weights.device)
  ess = sum_w.square() / (n * sum_w2).clamp_min(torch.finfo(weights.dtype).tiny)
  return torch.where(sum_w2 > 0, ess, torch.zeros_like(ess))


def bootstrap_time_outs(
  rewards: torch.Tensor,
  time_outs: torch.Tensor,
  values: torch.Tensor,
  gamma: float,
) -> torch.Tensor:
  """Add ``gamma * V(s_t)`` on truncated steps, mirroring RSL-RL's PPO.

  RSL-RL bootstraps time-outs with ``V(s_t)`` because auto-reset hides the true
  next state. MPOPI re-applies it with the current critic instead of reusing the
  bootstrap baked in at collection time.
  """
  return rewards + gamma * values * time_outs.to(values.dtype)


def vtrace(
  rewards: torch.Tensor,
  dones: torch.Tensor,
  values: torch.Tensor,
  bootstrap_values: torch.Tensor,
  log_ratio: torch.Tensor,
  gamma: float,
  lam: float,
  rho_clip: float | None = 1.0,
  trace_clip: float = 1.0,
  log_ratio_clamp: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """V-trace value targets and truncated-IS GAE advantages for one segment.

  Recursion (backwards over ``t``), with ``rho_t = pi_old / mu``::

    delta_t = r_t + gamma (1 - d_t) V(s_{t+1}) - V(s_t)
    v_t     = V(s_t) + min(rho_bar, rho_t) delta_t
              + gamma lam (1 - d_t) min(c_bar, rho_t) (v_{t+1} - V(s_{t+1}))
    A_t     = delta_t + gamma lam (1 - d_t) (v_{t+1} - V(s_{t+1}))

  With ``rho == 1`` and both clips ``>= 1`` this reduces exactly to GAE(lambda).

  Args:
    rewards: ``[T, N, 1]`` rewards, time-out bootstrap already applied.
    dones: ``[T, N, 1]`` terminated-or-truncated flags.
    values: ``[T, N, 1]`` critic values ``V(s_t)``.
    bootstrap_values: ``[N, 1]`` critic value of the state after the segment.
    log_ratio: ``[T, N, 1]`` ``log(pi_old / mu)``. Non-finite entries are
      treated as ratio 0 (trace cut); the caller must mask those samples.

  Returns:
    ``(returns, advantages)``, both ``[T, N, 1]``.
  """
  finite = torch.isfinite(log_ratio)
  safe = torch.where(finite, log_ratio, torch.zeros_like(log_ratio))
  rhos = torch.exp(safe.clamp(-log_ratio_clamp, log_ratio_clamp))
  rhos = torch.where(finite, rhos, torch.zeros_like(rhos))
  rho_bar = rhos.clamp(max=rho_clip)
  cs = rhos.clamp(max=trace_clip)
  not_done = 1.0 - dones.float()

  num_steps = rewards.shape[0]
  returns = torch.zeros_like(values)
  advantages = torch.zeros_like(values)
  # acc holds v_{t+1} - V(s_{t+1}); zero past the end of the segment.
  acc = torch.zeros_like(bootstrap_values)
  for t in reversed(range(num_steps)):
    next_values = bootstrap_values if t == num_steps - 1 else values[t + 1]
    delta = rewards[t] + not_done[t] * gamma * next_values - values[t]
    advantages[t] = delta + not_done[t] * gamma * lam * acc
    acc = rho_bar[t] * delta + not_done[t] * gamma * lam * cs[t] * acc
    returns[t] = values[t] + acc
  return returns, advantages
