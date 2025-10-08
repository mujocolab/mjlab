"""Abstract interface for debug visualization across different viewers."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

import numpy as np
import torch


class DebugVisualizer(Protocol):
  """Protocol for viewer-agnostic debug visualization.

  This allows command terms and other components to draw debug visualizations
  without knowing the underlying viewer implementation (MuJoCo native vs Viser).
  """

  @abstractmethod
  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Add an arrow from start to end position.

    Args:
      start: Start position (3D vector)
      end: End position (3D vector)
      color: RGBA color (values 0-1)
      width: Arrow shaft width
      label: Optional label for this arrow
    """
    pass

  @abstractmethod
  def add_ghost_mesh(
    self,
    qpos: np.ndarray | torch.Tensor,
    model: object | None = None,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost/transparent rendering of a robot at a target pose.

    Args:
      qpos: Joint positions for the ghost pose
      model: Model defining the robot geometry (optional, visualizer may use its own)
      alpha: Transparency (0=transparent, 1=opaque)
      label: Optional label for this ghost
    """
    pass

  @abstractmethod
  def clear(self) -> None:
    """Clear all debug visualizations."""
    pass


class NullDebugVisualizer:
  """No-op visualizer when visualization is disabled."""

  def add_arrow(self, start, end, color, width=0.015, label=None) -> None:
    pass

  def add_ghost_mesh(self, qpos, model, alpha=0.5, label=None) -> None:
    pass

  def clear(self) -> None:
    pass
