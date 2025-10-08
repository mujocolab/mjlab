"""MuJoCo native viewer debug visualizer implementation."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

if TYPE_CHECKING:
  pass


class MujocoNativeDebugVisualizer:
  """Debug visualizer for MuJoCo's native viewer.

  This implementation directly adds geometry to the MuJoCo scene using mjv_addGeoms
  and other MuJoCo visualization primitives.
  """

  def __init__(self, scn: mujoco.MjvScene, mj_model: mujoco.MjModel):
    """Initialize the MuJoCo native visualizer.

    Args:
      scn: MuJoCo scene to add visualizations to
      mj_model: MuJoCo model (used for ghost mesh rendering)
    """
    self.scn = scn
    self.mj_model = mj_model
    # Store the initial geom count to know where debug vis starts
    self._initial_geom_count = scn.ngeom

  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Add an arrow visualization using MuJoCo's arrow geometry."""
    # Convert to numpy if needed
    if isinstance(start, torch.Tensor):
      start = start.cpu().numpy()
    if isinstance(end, torch.Tensor):
      end = end.cpu().numpy()

    # Add new geom to scene
    self.scn.ngeom += 1
    geom = self.scn.geoms[self.scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    # Initialize as arrow
    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      size=np.array([0.005, 0.02, 0.02]),  # Arrow dimensions
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(color, dtype=np.float32),
    )

    # Set arrow endpoints
    mujoco.mjv_connector(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      width=width,
      from_=start,
      to=end,
    )

  def add_ghost_mesh(
    self,
    qpos: np.ndarray | torch.Tensor,
    model: object | None = None,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost mesh by rendering the robot at a different pose.

    This creates a semi-transparent copy of the robot geometry at the target pose.
    """
    # Convert to numpy if needed
    if isinstance(qpos, torch.Tensor):
      qpos = qpos.cpu().numpy()

    # Create visualization model and data if not provided
    if not hasattr(self, "_viz_model"):
      self._viz_model = copy.deepcopy(self.mj_model)
      # Make the ghost slightly more visible/different
      self._viz_model.geom_rgba[:, 1] = np.clip(
        self._viz_model.geom_rgba[:, 1] * 1.5, 0.0, 1.0
      )
      self._viz_data = mujoco.MjData(self._viz_model)
      self._vopt = mujoco.MjvOption()
      self._vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
      self._pert = mujoco.MjvPerturb()

    # Set the pose
    self._viz_data.qpos[:] = qpos

    # Forward kinematics
    mujoco.mj_forward(self._viz_model, self._viz_data)

    # Add to scene
    mujoco.mjv_addGeoms(
      self._viz_model,
      self._viz_data,
      self._vopt,
      self._pert,
      mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
      self.scn,
    )

  def clear(self) -> None:
    """Clear debug visualizations by resetting geom count."""
    # Reset to the initial geom count (before any debug vis was added)
    self.scn.ngeom = self._initial_geom_count
