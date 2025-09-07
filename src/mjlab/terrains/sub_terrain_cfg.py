from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import trimesh


@dataclass
class SubTerrainBaseCfg:
  """Base class for terrain configurations."""

  function: Callable[
    [float, SubTerrainBaseCfg], tuple[list[trimesh.Trimesh], np.ndarray]
  ]
  proportion: float = 1.0
  size: tuple[float, float] = (10.0, 10.0)
