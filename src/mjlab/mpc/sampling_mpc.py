"""Sampling-based MPC (MPPI / MPOPI) running on a batched mjlab planning env.

The planner keeps a second copy of the task with ``num_real * num_samples``
worlds. At every control step it copies the state of each real env into its
``num_samples`` planning worlds, rolls out ``num_samples`` perturbed action
sequences for ``horizon`` steps with the task's own reward manager, and
combines them with path-integral (MPPI) weights. ``iterations > 1`` adapts the
sampling distribution (mean and per-dimension std) between batches within the
same control step (MPOPI); ``iterations == 1`` is plain MPPI.

Actions live in the policy action space (before the action manager scales
them), so MPC actions and PPO actions are directly comparable.
"""

import contextlib
import copy
import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.mpc.config import SamplingMpcCfg
from mjlab.mpc.state_sync import freeze_commands, sync_env_state


@dataclass
class MpcPlan:
  action: torch.Tensor
  """``[N, A]`` first action of the optimized plan (noise-free)."""
  std: torch.Tensor
  """``[N, A]`` std of the final sampling distribution. Suitable as the
  execution-noise std when a known behavior density is needed."""
  best_return: torch.Tensor
  """``[N]`` best sampled return in the last iteration."""
  ess: torch.Tensor
  """``[N]`` normalized ESS of the final MPPI weights."""


@contextlib.contextmanager
def preserve_global_rng():
  """mjlab env construction reseeds Python, NumPy and torch globally."""
  states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
  cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
  try:
    yield
  finally:
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if cuda is not None:
      torch.cuda.set_rng_state_all(cuda)


def mppi_weights(
  returns: torch.Tensor,
  target_ess: float | None,
  temperature: float = 0.1,
  bisection_steps: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Path-integral weights ``w_k ∝ exp((R_k - max R) / λ)`` per row.

  With ``target_ess`` set, λ is chosen per row by bisection on ``log λ`` so
  that the normalized ESS ``(Σw)² / (K Σw²)`` equals ``target_ess``. ESS grows
  monotonically with λ, from 1/K (argmax) to 1 (uniform).

  Args:
    returns: ``[N, K]`` returns to maximize.

  Returns:
    ``(weights [N, K], normalized ESS [N])``.
  """
  num_samples = returns.shape[1]
  advantage = returns - returns.max(dim=1, keepdim=True).values  # <= 0.

  def weights_for(log_lam: torch.Tensor) -> torch.Tensor:
    return torch.softmax(advantage / log_lam.exp()[:, None], dim=1)

  def ess(w: torch.Tensor) -> torch.Tensor:
    return 1.0 / (num_samples * w.square().sum(dim=1))

  if target_ess is None:
    lam = torch.full_like(returns[:, 0], math.log(temperature))
    w = weights_for(lam)
    return w, ess(w)

  target = min(max(target_ess, 1.0 / num_samples), 1.0)
  lo = torch.full_like(returns[:, 0], -20.0)
  hi = torch.full_like(returns[:, 0], 20.0)
  for _ in range(bisection_steps):
    mid = 0.5 * (lo + hi)
    too_greedy = ess(weights_for(mid)) < target
    lo = torch.where(too_greedy, mid, lo)
    hi = torch.where(too_greedy, hi, mid)
  w = weights_for(0.5 * (lo + hi))
  return w, ess(w)


class SamplingMpc:
  """MPPI / MPOPI planner for ``num_real`` real envs of an mjlab task."""

  def __init__(
    self,
    env_cfg: ManagerBasedRlEnvCfg,
    num_real: int,
    cfg: SamplingMpcCfg,
    device: str = "cpu",
  ) -> None:
    """
    Args:
      env_cfg: Config of the task to plan on (copied, not modified). Use the
        training config: the planner must see the same dynamics and rewards.
      num_real: Number of real envs the planner controls.
      cfg: Planner config.
      device: Torch / simulation device.
    """
    if cfg.iterations < 1 or cfg.num_samples < 2 or cfg.horizon < 1:
      raise ValueError("Need iterations >= 1, num_samples >= 2 and horizon >= 1.")
    if cfg.num_knots is not None and cfg.num_knots < 1:
      raise ValueError("num_knots must be >= 1.")
    self.cfg = cfg
    self.num_real = num_real
    self.device = torch.device(device)
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.scene.num_envs = num_real * cfg.num_samples
    env_cfg.auto_reset = False
    env_cfg.terminations = {}  # Every sample is rolled out for the full horizon.
    # Random pushes would perturb samples unpredictably; keep startup events so
    # the same model fields are expanded per world as in the real env.
    env_cfg.events = {k: v for k, v in env_cfg.events.items() if v.mode != "interval"}
    with preserve_global_rng():
      self.env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
      self.env.reset()
    self.action_dim = self.env.action_manager.total_action_dim
    self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
    self.plan_seq = torch.zeros(
      num_real, cfg.horizon, self.action_dim, device=self.device
    )
    self.last_plan: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Forget the warm-start plan (e.g. after the real env was reset)."""
    if env_ids is None:
      self.plan_seq.zero_()
    else:
      self.plan_seq[env_ids] = 0.0

  def plan(self, real_env: ManagerBasedRlEnv) -> MpcPlan:
    """Optimize the plan from the real envs' current state."""
    cfg = self.cfg
    n, k, h, a = self.num_real, cfg.num_samples, cfg.horizon, self.action_dim
    mean = self.plan_seq.clone()
    std = torch.full_like(mean, cfg.noise_std)
    min_std, max_std = (
      cfg.min_std_scale * cfg.noise_std,
      cfg.max_std_scale * cfg.noise_std,
    )
    returns: torch.Tensor | None = None
    ess: torch.Tensor | None = None
    for it in range(cfg.iterations):
      noise = self._sample_noise(n, k, h, a)
      noise[:, 0] = 0.0  # Nominal sample.
      samples = mean[:, None] + std[:, None] * noise
      if cfg.action_clip is not None:
        samples = samples.clamp(-cfg.action_clip, cfg.action_clip)
      returns = self._rollout(real_env, samples)
      weights, ess = mppi_weights(returns, cfg.target_ess, cfg.temperature)
      w = weights[:, :, None, None]
      new_mean = (w * samples).sum(dim=1)
      if it < cfg.iterations - 1:
        # MPOPI: move and reshape the sampling distribution before the next batch.
        var = (w * (samples - new_mean[:, None]).square()).sum(dim=1)
        blended = (1 - cfg.std_smoothing) * std + cfg.std_smoothing * var.sqrt()
        std = blended.clamp(min_std, max_std)
      mean = new_mean
    assert returns is not None and ess is not None
    best_return = returns.max(dim=1).values
    first = mean[:, 0].clone()
    self.last_plan = mean.clone()
    # Warm start: shift the plan one step and repeat the last action.
    self.plan_seq = torch.cat([mean[:, 1:], mean[:, -1:]], dim=1)
    return MpcPlan(
      action=first, std=std[:, 0].clone(), best_return=best_return, ess=ess
    )

  def _sample_noise(self, n: int, k: int, h: int, a: int) -> torch.Tensor:
    """Standard normal noise ``[N, K, H, A]``, optionally smooth in time."""
    knots = self.cfg.num_knots
    if knots is None or knots >= h:
      return torch.randn(n, k, h, a, device=self.device, generator=self.generator)
    values = torch.randn(n, k, knots, a, device=self.device, generator=self.generator)
    if knots == 1:
      return values.expand(n, k, h, a).clone()
    # Linear interpolation between knots at t = 0, ..., h - 1.
    pos = torch.linspace(0, knots - 1, h, device=self.device)
    lo = pos.floor().long().clamp(max=knots - 2)
    frac = (pos - lo)[None, None, :, None]
    return values[:, :, lo] * (1 - frac) + values[:, :, lo + 1] * frac

  def _copy_state(self, real_env: ManagerBasedRlEnv) -> None:
    with torch.inference_mode():
      sync_env_state(real_env, self.env, self.cfg.num_samples)
      freeze_commands(self.env)

  def compare_with_policy(
    self,
    real_env: ManagerBasedRlEnv,
    policy: Callable[[TensorDict], torch.Tensor],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizon returns of the last plan and of ``policy`` from the same state.

    Call right after :meth:`plan` on the same ``real_env`` state. From each real
    env's state, planning world 0 executes the plan just optimized (open loop)
    and planning world 1 is driven by ``policy`` (closed loop on the planning
    env's observations), both for ``horizon`` steps with the task's rewards.

    Returns:
      ``(teacher_return [N], policy_return [N])``.
    """
    if self.last_plan is None:
      raise RuntimeError("compare_with_policy() needs a plan(): call plan() first.")
    n, k, h = self.num_real, self.cfg.num_samples, self.cfg.horizon
    self._copy_state(real_env)
    total = torch.zeros(n, k, device=self.device)
    with torch.inference_mode():
      obs = self.env.observation_manager.compute()
      for t in range(h):
        action = self.last_plan[:, t].repeat_interleave(k, dim=0).view(n, k, -1)
        policy_action = policy(TensorDict(obs, batch_size=[n * k])).view(n, k, -1)
        action = action.clone()
        action[:, 1] = policy_action[:, 1]
        obs, _, _, _, _ = self.env.step(action.view(n * k, -1))
        total += self.env.reward_buf.view(n, k)
    return total[:, 0], total[:, 1]

  def _rollout(
    self, real_env: ManagerBasedRlEnv, samples: torch.Tensor
  ) -> torch.Tensor:
    """Sum of rewards of each ``[N, K, H, A]`` action sequence -> ``[N, K]``."""
    n, k, h, a = samples.shape
    self._copy_state(real_env)
    total = torch.zeros(n * k, device=self.device)
    flat = samples.reshape(n * k, h, a)
    with torch.inference_mode():
      for t in range(h):
        self.env.step(flat[:, t])
        total += self.env.reward_buf
    return total.view(n, k)

  def close(self) -> None:
    self.env.close()
