# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, ParamSpec, TypeVar

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


##
# Observation manager.
##


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
  """Configuration for an observation term."""

  noise: NoiseCfg | NoiseModelCfg | None = None
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

  func: Any
  weight: float


##
# Termination manager.
##


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
  """Configuration for a termination term."""

  time_out: bool = False
  """Whethher the term contributes towards episodic timeouts. Defaults to False."""
