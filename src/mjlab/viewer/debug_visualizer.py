"""Abstract interface for debug visualization across different viewers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
  import mujoco


class DebugVisualizer(ABC):
  """Abstract base class for viewer-agnostic debug visualization.

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
      start: Start position (3D vector).
      end: End position (3D vector).
      color: RGBA color (values 0-1).
      width: Arrow shaft width.
      label: Optional label for this arrow.
    """
    ...

  @abstractmethod
  def add_ghost_mesh(
    self,
    qpos: np.ndarray | torch.Tensor,
    model: mujoco.MjModel,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost/transparent rendering of a robot at a target pose.

    Args:
      qpos: Joint positions for the ghost pose.
      model: MuJoCo model with pre-configured appearance (geom_rgba for colors).
      alpha: Transparency override (0=transparent, 1=opaque). May not be supported by
        all implementations.
      label: Optional label for this ghost.
    """
    ...

  @abstractmethod
  def add_frame(
    self,
    position: np.ndarray | torch.Tensor,
    rotation_matrix: np.ndarray | torch.Tensor,
    scale: float = 0.3,
    label: str | None = None,
    axis_radius: float = 0.01,
    alpha: float = 1.0,
    axis_colors: tuple[tuple[float, float, float], ...] | None = None,
  ) -> None:
    """Add a coordinate frame visualization.

    Args:
      position: Position of the frame origin (3D vector).
      rotation_matrix: Rotation matrix (3x3).
      scale: Scale/length of the axis arrows.
      label: Optional label for this frame.
      axis_radius: Radius/thickness of the axis arrows.
      alpha: Transparency override (0=transparent, 1=opaque). Note: The Viser
        implementation does not support per-arrow transparency. All arrows in the
        scene will share the same alpha value.
      axis_colors: Optional tuple of 3 RGB colors for X, Y, Z axes. Each color is a
        tuple of 3 floats (R, G, B) with values 0-1. If None, uses default RGB coloring
        (X=red, Y=green, Z=blue).
    """
    ...

  @abstractmethod
  def clear(self) -> None:
    """Clear all debug visualizations."""
    ...


class NullDebugVisualizer:
  """No-op visualizer when visualization is disabled."""

  def __init__(self, env_idx: int = 0):
    self.env_idx = env_idx

  def add_arrow(self, start, end, color, width=0.015, label=None) -> None:
    pass

  def add_ghost_mesh(self, qpos, model, alpha=0.5, label=None) -> None:
    pass

  def add_frame(
    self,
    position,
    rotation_matrix,
    scale=0.3,
    label=None,
    axis_radius=0.01,
    alpha=1.0,
    axis_colors=None,
  ) -> None:
    pass

  def clear(self) -> None:
    pass
