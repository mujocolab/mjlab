"""Configuration for the experimental MPOPI off-policy correction layer."""

import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from mjlab.mpc.config import SamplingMpcCfg

MpopiMode = Literal["ppo", "naive_replay_ppo", "mpopi_ppo", "mpc_ppo"]


@dataclass
class MpcDataCfg:
  """MPC-generated data for PPO (mode ``"mpc_ppo"``).

  A separate set of ``num_envs`` envs is driven by a sampling MPC controller
  with Gaussian execution noise, so the behavior density ``mu`` is known. Each
  PPO update adds the MPC segments in the buffer, importance-corrected by
  MPOPI, plus an optional behavior-cloning term toward the MPC action that is
  annealed to zero. Collection stops after ``collect_iterations`` so training
  ends as plain PPO. Defaults are placeholders without empirical backing.
  """

  num_envs: int = 16
  """Real envs driven by the MPC controller."""
  num_steps: int = 24
  """Control steps per collected segment."""
  collect_every: int = 1
  """Collect one segment every N PPO iterations."""
  collect_iterations: int = 50
  """Stop collecting after this many PPO iterations."""
  execution_std: float = 0.3
  """Std of the executed Gaussian noise around the MPC action, in policy
  action units. Defines ``mu(a|s) = N(a; u0(s), execution_std^2)``. 0 executes
  the MPC action itself; there is then no behavior density, so the data can
  only be used for behavior cloning (``use_in_ppo=False``)."""
  driver: Literal["mpc", "policy"] = "mpc"
  """Who acts in the MPC envs. ``"policy"`` is DAgger: the current policy acts
  and the MPC only labels the states it visits, so behavior cloning happens on
  the policy's own state distribution. Requires ``use_in_ppo=False``."""
  buffer_segments: int = 8
  """MPC segments kept in the buffer (all of them are used at every update)."""
  max_age: int | None = 10
  """Segments collected more than this many iterations ago are not used."""
  use_in_ppo: bool = True
  """Add MPC samples to PPO's surrogate and value losses."""
  inject_fraction: float | None = None
  """Fraction of each PPO batch made of MPC samples, drawn uniformly from the
  buffer at every update (MPC-Injection with a fixed mixing ratio). None uses
  every MPC sample in the buffer."""
  correction: bool = True
  """Importance-correct MPC samples (MPOPI). False treats them as on-policy."""
  bc_coef: float = 1.0
  """Initial weight of the behavior-cloning loss ``||mean_pi(s) - u0(s)||^2``
  (mean over action dims) on MPC samples. 0 disables it."""
  bc_iterations: int = 50
  """The behavior-cloning weight decays linearly over this many iterations."""
  bc_floor: float = 0.0
  """Lower bound of the behavior-cloning weight after the decay. Above 0 the
  policy keeps a small pull toward the MPC for the whole run, which needs the
  MPC data to stay in the buffer (``max_age=None``)."""
  planner: SamplingMpcCfg = field(default_factory=SamplingMpcCfg)

  def bc_weight(self, iteration: int) -> float:
    if self.bc_coef == 0.0:
      return 0.0
    if self.bc_iterations <= 0:
      return self.bc_floor
    decayed = self.bc_coef * max(0.0, 1.0 - iteration / self.bc_iterations)
    return max(self.bc_floor, decayed)

  def collects(self, iteration: int) -> bool:
    return iteration < self.collect_iterations and iteration % self.collect_every == 0

  def validate(self) -> None:
    if self.num_envs < 1 or self.num_steps < 1 or self.collect_every < 1:
      raise ValueError("num_envs, num_steps and collect_every must be >= 1.")
    if self.execution_std < 0.0:
      raise ValueError("execution_std must be >= 0.")
    if self.use_in_ppo and self.driver == "policy":
      raise ValueError(
        "DAgger labels are not the executed actions: use use_in_ppo=False."
      )
    if self.use_in_ppo and self.correction and self.execution_std == 0.0:
      raise ValueError(
        "Importance correction needs a behavior density: use execution_std > 0,"
        " correction=False (naive injection) or use_in_ppo=False."
      )
    if self.inject_fraction is not None and not 0.0 < self.inject_fraction < 1.0:
      raise ValueError("inject_fraction must be in (0, 1).")
    if not 0.0 <= self.bc_floor <= self.bc_coef:
      raise ValueError("bc_floor must be in [0, bc_coef].")
    if self.buffer_segments < 1:
      raise ValueError("buffer_segments must be >= 1.")
    if self.max_age is not None and self.max_age < 0:
      raise ValueError("max_age must be >= 0.")
    if self.bc_coef < 0.0:
      raise ValueError("bc_coef must be >= 0.")


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
  - ``"mpc_ppo"``: no replay of PPO's own data; MPC-generated data (see
    ``mpc``) is added instead, importance-corrected with the same estimators.
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
  mpc: MpcDataCfg = field(default_factory=MpcDataCfg)
  """MPC data source; only used in mode ``"mpc_ppo"``."""

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "MpopiCfg":
    """Rebuild from ``dataclasses.asdict`` output (nested dicts included)."""
    data = dict(data)
    mpc = data.pop("mpc", None)
    if isinstance(mpc, dict):
      mpc = dict(mpc)
      planner = mpc.pop("planner", None)
      if isinstance(planner, dict):
        mpc["planner"] = SamplingMpcCfg(**planner)
      data["mpc"] = MpcDataCfg(**mpc)
    elif mpc is not None:
      data["mpc"] = mpc
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
      raise TypeError(f"Unknown MpopiCfg fields: {sorted(unknown)}")
    return cls(**data)

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
    if self.mode == "mpc_ppo":
      self.mpc.validate()
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
