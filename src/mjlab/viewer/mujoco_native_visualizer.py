"""MuJoCo native viewer debug visualizer implementation."""

from __future__ import annotations

import mujoco
import numpy as np
import torch


class MujocoNativeDebugVisualizer:
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

  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Add an arrow visualization using MuJoCo's arrow geometry."""
    if isinstance(start, torch.Tensor):
      start = start.cpu().numpy()
    if isinstance(end, torch.Tensor):
      end = end.cpu().numpy()

    # Add new geom to scene.
    self.scn.ngeom += 1
    geom = self.scn.geoms[self.scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    # Initialize as arrow.
    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      size=np.array([0.005, 0.02, 0.02]),  # Arrow dimensions.
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(color, dtype=np.float32),
    )

    # Set arrow endpoints.
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

  def clear(self) -> None:
    """Clear debug visualizations by resetting geom count."""
    # Reset to the initial geom count (before any debug vis was added)
    self.scn.ngeom = self._initial_geom_count
