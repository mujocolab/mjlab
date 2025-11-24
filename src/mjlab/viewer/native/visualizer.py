"""MuJoCo native viewer debug visualizer implementation."""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from typing_extensions import override

from mjlab.viewer.debug_visualizer import DebugVisualizer


class MujocoNativeDebugVisualizer(DebugVisualizer):
  """Debug visualizer for MuJoCo's native viewer.

  This implementation directly adds geometry to the MuJoCo scene using mjv_addGeoms
  and other MuJoCo visualization primitives.
  """

  def __init__(self, scn: mujoco.MjvScene, mj_model: mujoco.MjModel, env_idx: int):
    """Initialize the MuJoCo native visualizer.

    Args:
      scn: MuJoCo scene to add visualizations to
      mj_model: MuJoCo model for creating visualization data
      env_idx: Index of the environment being visualized
    """
    self.scn = scn
    self.mj_model = mj_model
    self.env_idx = env_idx
    self._initial_geom_count = scn.ngeom

    self._vopt = mujoco.MjvOption()
    self._vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    self._pert = mujoco.MjvPerturb()
    self._viz_data = mujoco.MjData(mj_model)

  @override
  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Add an arrow visualization using MuJoCo's arrow geometry."""
    del label  # Unused.

    if isinstance(start, torch.Tensor):
      start = start.cpu().numpy()
    if isinstance(end, torch.Tensor):
      end = end.cpu().numpy()

    self.scn.ngeom += 1
    geom = self.scn.geoms[self.scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      width=width,
      from_=start,
      to=end,
    )

  @override
  def add_ghost_mesh(
    self,
    qpos: np.ndarray | torch.Tensor,
    model: mujoco.MjModel,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost mesh by rendering the robot at a different pose.

    This creates a semi-transparent copy of the robot geometry at the target pose.

    Args:
      qpos: Joint positions for the ghost pose
      model: MuJoCo model with pre-configured appearance (geom_rgba for colors)
      alpha: Transparency override (not used in MuJoCo implementation)
      label: Optional label (not used in MuJoCo implementation)
    """
    del alpha, label  # Unused.

    if isinstance(qpos, torch.Tensor):
      qpos = qpos.cpu().numpy()

    self._viz_data.qpos[:] = qpos
    mujoco.mj_forward(model, self._viz_data)

    mujoco.mjv_addGeoms(
      model,
      self._viz_data,
      self._vopt,
      self._pert,
      mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
      self.scn,
    )

  @override
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
    """Add a coordinate frame visualization with RGB-colored axes.

    This implementation reuses add_arrow to draw the three axis arrows.

    Args:
      position: Position of the frame origin (3D vector)
      rotation_matrix: Rotation matrix (3x3)
      scale: Scale/length of the axis arrows
      label: Optional label for this frame.
      axis_radius: Radius of the axis arrows.
      alpha: Opacity for all axes (0=transparent, 1=opaque).
      axis_colors: Optional tuple of 3 RGB colors for X, Y, Z axes. If None, uses
        default RGB coloring (X=red, Y=green, Z=blue).
    """
    del label  # Unused.

    if isinstance(position, torch.Tensor):
      position = position.cpu().numpy()
    if isinstance(rotation_matrix, torch.Tensor):
      rotation_matrix = rotation_matrix.cpu().numpy()

    default_colors = [(0.9, 0, 0), (0, 0.9, 0.0), (0.0, 0.0, 0.9)]
    colors = axis_colors if axis_colors is not None else default_colors

    for axis_idx in range(3):
      axis_direction = rotation_matrix[:, axis_idx]
      end_position = position + axis_direction * scale
      rgb = colors[axis_idx]
      color_rgba = (rgb[0], rgb[1], rgb[2], alpha)
      self.add_arrow(
        start=position,
        end=end_position,
        color=color_rgba,
        width=axis_radius,
      )

  @override
  def clear(self) -> None:
    """Clear debug visualizations by resetting geom count."""
    self.scn.ngeom = self._initial_geom_count
