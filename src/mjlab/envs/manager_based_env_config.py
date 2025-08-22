from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

from mjlab.scene.scene_config import SceneCfg
from mjlab.sim.sim_config import SimulationCfg
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import term
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.envs.types import T_observations, T_actions, T_events


@dataclass
class DefaultEventManagerCfg:
  reset_scene_to_default: EventTerm = term(
    EventTerm,
    func=reset_scene_to_default,
    mode="reset",
  )


@dataclass(kw_only=True)
class ManagerBasedEnvCfg(Generic[T_observations, T_actions, T_events]):
  decimation: int
  scene: SceneCfg
  observations: T_observations
  actions: T_actions
  events: T_events = field(default_factory=DefaultEventManagerCfg)  # type: ignore
  seed: int | None = None
  sim: SimulationCfg = field(default_factory=SimulationCfg)
