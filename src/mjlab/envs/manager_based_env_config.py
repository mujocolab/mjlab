from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.scene.scene_config import SceneCfg
from mjlab.sim.sim_config import SimulationCfg
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import term
from mjlab.envs.mdp.events import reset_scene_to_default


@dataclass
class DefaultEventManagerCfg:
  reset_scene_to_default: EventTerm = term(
    EventTerm,
    func=reset_scene_to_default,
    mode="reset",
  )


@dataclass(kw_only=True)
class ManagerBasedEnvCfg:
  decimation: int
  scene: SceneCfg
  observations: object
  actions: object
  events: DefaultEventManagerCfg = field(default_factory=DefaultEventManagerCfg)
  seed: int | None = None
  sim: SimulationCfg = field(default_factory=SimulationCfg)
