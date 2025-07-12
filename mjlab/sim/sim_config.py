from dataclasses import dataclass
from mjlab.entities.common.config import OptionCfg


@dataclass(frozen=True)
class SimulationCfg:
  num_envs: int = 1
  nconmax: int | None = None
  njmax: int | None = None
  device: str = "cuda:0"
  mujoco: OptionCfg = OptionCfg()
