from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, ParamSpec, TypeVar

import torch

from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.command_manager import CommandTerm
from mjlab.utils.noise.noise_cfg import NoiseCfg, NoiseModelCfg

P = ParamSpec("P")
T = TypeVar("T")


def term(term_cls: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
  return field(default_factory=lambda: term_cls(*args, **kwargs))


@dataclass
class ManagerTermBaseCfg:
  func: Any
  params: dict[str, Any] = field(default_factory=lambda: {})


##
# Action manager.
##


@dataclass(kw_only=True)
class ActionTermCfg:
  """Configuration for an action term."""

  class_type: type[ActionTerm]
  asset_name: str
  clip: dict[str, tuple] | None = None


##
# Command manager.
##


@dataclass(kw_only=True)
class CommandTermCfg:
  """Configuration for a command generator term."""

  class_type: type[CommandTerm]
  resampling_time_range: tuple[float, float]
  debug_vis: bool = False


##
# Curriculum manager.
##


@dataclass(kw_only=True)
class CurriculumTermCfg(ManagerTermBaseCfg):
  pass


##
# Event manager.
##


EventMode = Literal["startup", "reset", "interval"]


@dataclass(kw_only=True)
class EventTermCfg(ManagerTermBaseCfg):
  """Configuration for an event term."""

  mode: EventMode
  interval_range_s: tuple[float, float] | None = None
  is_global_time: bool = False
  min_step_count_between_reset: int = 0
  domain_randomization: bool = False
  """Whether this event term performs domain randomization. If True, the field
  name (from params["field"]) will be tracked for domain randomization purposes."""


##
# Observation manager.
##


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
  """Configuration for an observation term.

  Processing pipeline: compute → noise → clip → scale → delay → history.
  Delay models sensor latency. History provides temporal context. Both are optional
  and can be combined.
  """

  noise: NoiseCfg | NoiseModelCfg | None = None
  """Noise model to apply to the observation."""
  clip: tuple[float, float] | None = None
  """Range (min, max) to clip the observation values."""
  scale: tuple[float, ...] | float | torch.Tensor | None = None
  """Scaling factor(s) to multiply the observation by."""
  delay_min_lag: int = 0
  """Minimum lag (in steps) for delayed observations. Lag sampled uniformly from
  [min_lag, max_lag]. Convert to ms: lag * (1000 / control_hz)."""
  delay_max_lag: int = 0
  """Maximum lag (in steps) for delayed observations. Use min=max for constant delay."""
  delay_per_env: bool = True
  """If True, each environment samples its own lag. If False, all environments share
  the same lag at each step."""
  delay_hold_prob: float = 0.0
  """Probability of reusing the previous lag instead of resampling. Useful for
  temporally correlated latency patterns."""
  delay_update_period: int = 0
  """Resample lag every N steps (models multi-rate sensors). If 0, update every step."""
  delay_per_env_phase: bool = True
  """If True and update_period > 0, stagger update timing across envs to avoid
  synchronized resampling."""
  history_length: int = 0
  """Number of past observations to keep in history. 0 = no history."""
  flatten_history_dim: bool = True
  """Whether to flatten the history dimension into observation.

  When True and concatenate_terms=True, uses term-major ordering:
  [A_t0, A_t1, ..., A_tH-1, B_t0, B_t1, ..., B_tH-1, ...]
  See docs/api/observation_history_delay.md for details on ordering."""


@dataclass
class ObservationGroupCfg:
  """Configuration for an observation group.

  The `terms` field contains a dictionary mapping term names to their configurations.
  """

  terms: dict[str, ObservationTermCfg]
  concatenate_terms: bool = True
  concatenate_dim: int = -1
  enable_corruption: bool = False
  history_length: int | None = None
  flatten_history_dim: bool = True


##
# Reward manager.
##


@dataclass(kw_only=True)
class RewardTermCfg(ManagerTermBaseCfg):
  """Configuration for a reward term."""

  func: Any
  weight: float


##
# Termination manager.
##


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
  """Configuration for a termination term."""

  time_out: bool = False
  """Whether the term contributes towards episodic timeouts."""
