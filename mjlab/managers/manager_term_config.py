from dataclasses import dataclass, field
from typing import Any, Callable, Type


def term(term_cls: Type, **kwargs) -> Any:
  return field(default_factory=lambda: term_cls(**kwargs))


@dataclass
class ManagerTermBaseCfg:
  func: Callable
  params: dict[str, Any] = field(default_factory=lambda: {})


##
# Event manager.
##


@dataclass
class EventTermCfg(ManagerTermBaseCfg):
  """Configuration for an event term."""

  mode: str
  interval_range_s: tuple[float, float] | None = None
  is_global_time: bool = False
  min_step_count_between_reset: int = 0


##
# Observation manager.
##


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
  """Configuration for an observation term."""

  noise: Any | None = None
  clip: tuple[float, float] | None = None
  history_length: int = 0
  flatten_history_dim: bool = True


@dataclass
class ObservationGroupCfg:
  """Configuration for an observation group."""

  concatenate_terms: bool = True
  concatenate_dim: int = -1
  enable_corruption: bool = False


##
# Reward manager.
##


@dataclass
class RewardTermCfg(ManagerTermBaseCfg):
  """Configuration for a reward term."""

  weight: float = 0.0


##
# Termination manager.
##


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
  """Configuration for a termination term."""

  time_out: bool = False
  """Whethher the term contributes towards episodic timeouts. Defaults to False."""
