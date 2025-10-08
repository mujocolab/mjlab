"""Abstract interface for debug visualization across different viewers."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

if TYPE_CHECKING:
  import mujoco


class DebugVisualizer(Protocol):
  """Protocol for viewer-agnostic debug visualization.

  This allows manager terms to draw debug visualizations without knowing the underlying
  viewer implementation.
  """

  env_idx: int
  """Index of the environment being visualized."""

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
    model: "mujoco.MjModel",
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost/transparent rendering of a robot at a target pose.

    Args:
      qpos: Joint positions for the ghost pose
      model: MuJoCo model with pre-configured appearance (geom_rgba for colors)
      alpha: Transparency override (0=transparent, 1=opaque) - may not be used by all implementations
      label: Optional label for this ghost
    """
    pass

  @abstractmethod
  def clear(self) -> None:
    """Clear all debug visualizations."""
    pass


class NullDebugVisualizer:
  """No-op visualizer when visualization is disabled."""

  def __init__(self, env_idx: int = 0):
    self.env_idx = env_idx

  def add_arrow(self, start, end, color, width=0.015, label=None) -> None:
    pass

  def add_ghost_mesh(self, qpos, model, alpha=0.5, label=None) -> None:
    pass

  def clear(self) -> None:
    pass
