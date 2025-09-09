from __future__ import annotations

import abc
from dataclasses import dataclass

import mujoco


@dataclass
class SubTerrainCfg(abc.ABC):
  proportion: float = 1.0
  size: tuple[float, float] = (10.0, 10.0)

  @abc.abstractmethod
  def function(self, difficulty: float, spec: mujoco.MjSpec):
    raise NotImplementedError
