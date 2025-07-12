from dataclasses import dataclass

from mjlab.entities.scene.scene_config import SceneCfg
from mjlab.sim.sim_config import SimulationCfg


@dataclass
class ManagerBasedEnvCfg:
  decimation: int
  scene: SceneCfg
  seed: int | None = None
  sim: SimulationCfg = SimulationCfg()
