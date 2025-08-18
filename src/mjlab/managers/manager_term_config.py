from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, Type, TypeVar

import torch

if TYPE_CHECKING:
  from mjlab.managers.action_manager import ActionTerm
  from mjlab.managers.command_manager import CommandTerm

P = ParamSpec("P")
T = TypeVar("T")


def term(term_cls: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
  return field(default_factory=lambda: term_cls(*args, **kwargs))


@dataclass
class ManagerTermBaseCfg:
  func: Callable
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


@dataclass(kw_only=True)
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


@dataclass
class ObservationGroupCfg:
  """Configuration for an observation group."""

  concatenate_terms: bool = True
  concatenate_dim: int = -1
  enable_corruption: bool = False


##
# Reward manager.
##


@dataclass(kw_only=True)
class RewardTermCfg(ManagerTermBaseCfg):
  """Configuration for a reward term."""

  func: Callable[..., torch.Tensor]
  weight: float

  # TODO(kevin): Sanity check weight is valid type.


##
# Termination manager.
##


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
  """Configuration for a termination term."""

  time_out: bool = False
  """Whethher the term contributes towards episodic timeouts. Defaults to False."""
