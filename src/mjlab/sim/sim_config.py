from dataclasses import dataclass, field

from mjlab.utils.spec_editor.spec_editor_config import OptionCfg


@dataclass(kw_only=True)
class RenderCfg:
  enable_reflections: bool = True
  enable_shadows: bool = True
  camera: str | int | None = -1
  height: int = 240
  width: int = 320


@dataclass(kw_only=True)
class SimulationCfg:
  num_envs: int = 1
  device: str = "cuda:0"
  nconmax: int | None = None
  njmax: int | None = None
  ls_parallel: bool = True  # Boosts perf quite noticeably.
  mujoco: OptionCfg = field(default_factory=OptionCfg)
  render: RenderCfg = field(default_factory=RenderCfg)
