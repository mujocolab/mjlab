"""Configuration for the experimental MPOPI off-policy correction layer."""

import warnings
from dataclasses import dataclass
from typing import Literal

MpopiMode = Literal["ppo", "naive_replay_ppo", "mpopi_ppo"]


@dataclass
class MpopiCfg:
  """Config for MPOPI, the replay correction stage that feeds PPO.

  See ``docs/source/training/mpopi.rst`` for the objective and estimator.
  """

  mode: MpopiMode = "ppo"
  """Training mode.

  - ``"ppo"``: standard RSL-RL PPO. MPOPI is fully disabled and the upstream
    ``PPO`` class receives exactly the same arguments as without MPOPI.
  - ``"naive_replay_ppo"``: replay data is added to PPO's batch as if it were
    on-policy (importance ratio forced to 1). Baseline for the correction.
  - ``"mpopi_ppo"``: replay data is importance-corrected by MPOPI.
  """
  replay_buffer_size: int = 4
  """Number of past rollout segments (iterations) kept in the replay buffer.
  Placeholder default without empirical backing; ablate it."""
  replay_ratio: float = 1.0
  """Replay samples added per fresh on-policy sample."""
  replay_batch_size: int | None = None
  """Absolute number of replay samples per update. Overrides ``replay_ratio``."""
  sampling_strategy: Literal["uniform", "all"] = "uniform"
  """``"uniform"`` samples replay transitions uniformly without replacement.
  ``"all"`` uses every eligible transition and ignores the replay size."""
  max_policy_age: int | None = None
  """Segments collected more than this many iterations ago are excluded.
  None keeps everything in the buffer."""
  max_abs_log_ratio: float | None = None
  """Reject samples with ``|log(pi_old / mu)|`` above this value (support /
  coverage filter). None disables the filter."""
  importance_weight_clip_min: float = 0.0
  """Lower clip on the surrogate importance weight. 0 disables it."""
  importance_weight_clip_max: float | None = 1.0
  """Truncation level (rho bar) on the surrogate weight and the V-trace TD term.
  1.0 follows V-trace / IMPALA. None disables truncation."""
  trace_clip_max: float = 1.0
  """Truncation level (c bar) on the V-trace trace coefficients."""
  log_ratio_clamp: float = 20.0
  """Numerical guard: log ratios are clamped to ``[-x, x]`` before ``exp``."""
  min_ess: float = 0.0
  """Minimum normalized effective sample size in ``[0, 1]``. When the replay
  batch falls below it, all replay samples are rejected for that update.
  0 disables the gate."""
  weight_normalization: Literal["none", "self_normalized"] = "none"
  """``"self_normalized"`` rescales accepted replay weights to mean 1."""

  @property
  def enabled(self) -> bool:
    return self.mode != "ppo"

  def validate(self) -> None:
    if self.replay_buffer_size < 1:
      raise ValueError("replay_buffer_size must be >= 1.")
    if self.replay_ratio < 0.0:
      raise ValueError("replay_ratio must be >= 0.")
    if self.replay_batch_size is not None and self.replay_batch_size < 0:
      raise ValueError("replay_batch_size must be >= 0.")
    if self.max_policy_age is not None and self.max_policy_age < 1:
      raise ValueError("max_policy_age must be >= 1 (replay age is always >= 1).")
    if not 0.0 <= self.min_ess <= 1.0:
      raise ValueError("min_ess must be in [0, 1].")
    if self.log_ratio_clamp <= 0.0:
      raise ValueError("log_ratio_clamp must be > 0.")
    if self.trace_clip_max <= 0.0:
      raise ValueError("trace_clip_max must be > 0.")
    clip_max = self.importance_weight_clip_max
    if clip_max is not None:
      if clip_max <= self.importance_weight_clip_min:
        raise ValueError(
          "importance_weight_clip_max must exceed importance_weight_clip_min."
        )
      if clip_max < self.trace_clip_max:
        warnings.warn(
          "V-trace assumes importance_weight_clip_max >= trace_clip_max.",
          stacklevel=2,
        )
