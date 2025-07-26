from dataclasses import dataclass, field
from mjlab.utils.spec_editor.spec_editor_config import OptionCfg


@dataclass
class MjWarpConfig:
  nconmax: int | None = None
  njmax: int | None = None


@dataclass(kw_only=True)
class SimulationCfg:
  num_envs: int = 1
  device: str = "cuda:0"
  nconmax: int | None = None
  njmax: int | None = None
  ls_parallel: bool = True  # Boosts perf quite noticeably.
  mujoco: OptionCfg = field(default_factory=OptionCfg)
