from __future__ import annotations

from dataclasses import dataclass, MISSING, field

from mjlab.entities.scene.scene_config import SceneCfg
from mjlab.sim.sim_config import SimulationCfg


@dataclass(kw_only=True)
class ManagerBasedEnvCfg:
  decimation: int = MISSING
  scene: SceneCfg = MISSING
  observations: object = MISSING
  actions: object = MISSING
  seed: int | None = None
  sim: SimulationCfg = field(default_factory=SimulationCfg)
