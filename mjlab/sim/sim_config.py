from dataclasses import dataclass, field
from mjlab.utils.spec_editor.spec_editor_config import OptionCfg


@dataclass(kw_only=True)
class SimulationCfg:
  num_envs: int = 1
  nconmax: int | None = None
  njmax: int | None = None
  device: str = "cuda:0"
  mujoco: OptionCfg = field(default_factory=OptionCfg)
