from __future__ import annotations

from typing import TypedDict

import torch

from .numerics import finite_or_default, safe_mean_finite, sanitize_to_range


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def reward_weight(
  env,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  del env_ids
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter >= stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight], device=env.device)


class reward_weight_by_metric_progress:
  def __init__(self, cfg, env):
    del cfg, env
    self._progress_ema: torch.Tensor | None = None

  def __call__(
    self,
    env,
    env_ids: torch.Tensor | slice,
    reward_name: str,
    metric_name: str,
    progress_min: float,
    progress_max: float,
    weight_min: float,
    weight_max: float,
    ema_alpha: float = 0.1,
    weight_lerp: float = 0.2,
    invert_metric: bool = False,
    min_steps_per_episode: int = 8,
  ) -> dict[str, torch.Tensor]:
    if progress_max <= progress_min:
      raise ValueError(
        f"progress_max ({progress_max}) must be > progress_min ({progress_min})."
      )

    ema_alpha = float(torch.clamp(torch.tensor(ema_alpha), min=0.0, max=1.0).item())
    weight_lerp = float(
      torch.clamp(torch.tensor(weight_lerp), min=0.0, max=1.0).item()
    )

    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)

    if metric_name not in env.metrics_manager._episode_sums:
      current_weight = finite_or_default(float(reward_term_cfg.weight), weight_min)
      reward_term_cfg.weight = current_weight
      return {
        "weight": torch.tensor(current_weight, device=env.device),
        "progress_ema": torch.tensor(0.0, device=env.device),
      }

    episode_sums = env.metrics_manager._episode_sums[metric_name][env_ids]
    step_counts = env.metrics_manager._step_count[env_ids].float()
    valid_mask = step_counts >= float(min_steps_per_episode)
    if not torch.any(valid_mask):
      current_weight = finite_or_default(float(reward_term_cfg.weight), weight_min)
      reward_term_cfg.weight = current_weight
      return {
        "weight": torch.tensor(current_weight, device=env.device),
        "progress_ema": (
          self._progress_ema
          if self._progress_ema is not None
          else torch.tensor(0.0, device=env.device)
        ),
      }

    metric_samples = episode_sums[valid_mask] / torch.clamp(
      step_counts[valid_mask], min=1.0
    )
    metric_avg = safe_mean_finite(metric_samples)
    if metric_avg is None:
      current_weight = finite_or_default(float(reward_term_cfg.weight), weight_min)
      reward_term_cfg.weight = current_weight
      return {
        "weight": torch.tensor(current_weight, device=env.device),
        "progress_ema": (
          self._progress_ema
          if self._progress_ema is not None
          else torch.tensor(0.0, device=env.device)
        ),
      }

    progress_raw = sanitize_to_range(
      (metric_avg - progress_min) / (progress_max - progress_min),
      0.0,
      1.0,
      nan_default=0.0,
    )
    if invert_metric:
      progress_raw = 1.0 - progress_raw

    if self._progress_ema is None or not torch.isfinite(self._progress_ema).item():
      self._progress_ema = progress_raw.detach()
    else:
      self._progress_ema = (
        (1.0 - ema_alpha) * self._progress_ema + ema_alpha * progress_raw.detach()
      )
    self._progress_ema = sanitize_to_range(
      self._progress_ema,
      0.0,
      1.0,
      nan_default=0.0,
    )

    target_weight = sanitize_to_range(
      weight_min + self._progress_ema * (weight_max - weight_min),
      weight_min,
      weight_max,
      nan_default=weight_min,
    )
    current_weight = torch.tensor(
      finite_or_default(float(reward_term_cfg.weight), float(target_weight.item())),
      device=env.device,
    )
    new_weight = sanitize_to_range(
      (1.0 - weight_lerp) * current_weight + weight_lerp * target_weight,
      weight_min,
      weight_max,
      nan_default=float(target_weight.item()),
    )
    reward_term_cfg.weight = float(new_weight.item())

    return {
      "metric": metric_avg.detach(),
      "progress_raw": progress_raw.detach(),
      "progress_ema": self._progress_ema.detach(),
      "target_weight": target_weight.detach(),
      "weight": new_weight.detach(),
    }
