"""Mjlab viewer based on Viser.

Adapted from an MJX visualizer by Chung Min Kim: https://github.com/chungmin99/
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import trimesh
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj  # type: ignore
from typing_extensions import override

from mjlab.sim.sim import Simulation
from mjlab.viewer.base import BaseViewer, EnvProtocol, PolicyProtocol, VerbosityLevel


class ViserViewer(BaseViewer):
  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    render_all_envs: bool = True,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
  ) -> None:
    super().__init__(env, policy, frame_rate, render_all_envs, verbosity)

  @override
  def setup(self) -> None:
    """Setup the viewer resources."""

    self._server = viser.ViserServer()
    self._handles: dict[str, tuple[viser.BatchedMeshHandle, bool]] = {}
    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._batch_size = self.env.num_envs
    self._env_offsets = np.array([1.0, 1.0, 0.0]) * np.random.normal(
      scale=3.0, size=(self._batch_size, 3)
    )
    self._counter = 0

    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)

    mj_model = sim.mj_model

    # Populate GUI.
    with self._server.gui.add_folder("Visibility"):
      cb_collision = self._server.gui.add_checkbox("Collision geom", initial_value=True)
      cb_visual = self._server.gui.add_checkbox("Visual geom", initial_value=True)
      cb_floor = self._server.gui.add_checkbox("Floor geom", initial_value=False)

      @cb_collision.on_update
      def _(_) -> None:
        # Floor name is hack.
        for name, (handle, is_collision) in self._handles.items():
          if is_collision and "floor" not in name:
            handle.visible = cb_collision.value

      @cb_visual.on_update
      def _(_) -> None:
        for handle, is_collision in self._handles.values():
          if not is_collision:
            handle.visible = cb_visual.value

      @cb_floor.on_update
      def _(_) -> None:
        for name, handle in self._handles.items():
          if name.startswith("floor"):
            handle[0].visible = cb_floor.value

    # Process each geom in the model.
    self._server.scene.add_grid("/floor", width=20.0, height=20.0)
    for i in range(mj_model.ngeom):
      # Get geom properties
      name = mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, i)
      if not name:
        name = f"geom_{i}"

      pos = mj_model.geom_pos[i]
      quat = mj_model.geom_quat[i]  # (w, x, y, z)

      # Set color based on whether it's a collision geom
      is_collision = mj_model.geom_contype[i] != 0 or mj_model.geom_conaffinity[i] != 0

      # Handle different geom types using Viser primitives where possible
      mesh = self._create_mesh(mj_model, i)
      handle = self._server.scene.add_batched_meshes_simple(
        f"/geoms/{name}",
        vertices=mesh.vertices,
        faces=mesh.faces,
        batched_colors=(200, 100, 100) if is_collision else (30, 125, 230),
        batched_wxyzs=quat[None].repeat(self._batch_size, axis=0),
        batched_positions=pos[None].repeat(self._batch_size, axis=0),
        opacity=0.5 if is_collision else 1.0,
        flat_shading=True,
        lod="auto",
        visible="floor" not in name,
      )

      self._handles[name] = (handle, is_collision)

  @override
  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""

    # Skip every other frame to reduce load. 30FPS is fine!
    self._counter += 1
    if self._counter % 2 == 0:
      return

    # We'll make a copy of the relevant state, then do the update itself asynchronously.
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    mj_model = sim.mj_model
    wp_data = sim.wp_data

    geom_xpos = wp_data.geom_xpos.numpy()
    assert geom_xpos.shape == (self._batch_size, mj_model.ngeom, 3)
    geom_xmat = wp_data.geom_xmat.numpy()
    assert geom_xmat.shape == (self._batch_size, mj_model.ngeom, 3, 3)

    def update_mujoco() -> None:
      with self._server.atomic():
        geom_xquat = vtf.SO3.from_matrix(geom_xmat).wxyz
        for i in range(mj_model.ngeom):
          name = mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, i)
          if not name:
            name = f"geom_{i}"
          if name not in self._handles:
            continue

          # Update position and orientation
          handle = self._handles[name][0]
          handle.batched_positions = geom_xpos[..., i, :] + self._env_offsets[:, :]
          handle.batched_wxyzs = geom_xquat[..., i, :]

    self._threadpool.submit(update_mujoco)

  @override
  def sync_viewer_to_env(self) -> None:
    """Synchronize viewer state to environment (e.g., perturbations)."""
    # Does nothing for Viser.
    pass

  @override
  def close(self) -> None:
    """Close the viewer and cleanup resources."""
    pass

  @staticmethod
  def _create_mesh(mj_model, idx: int) -> trimesh.Trimesh:
    """
    Create a trimesh object from a geom in the MuJoCo model.
    """
    size = mj_model.geom_size[idx]
    geom_type = mj_model.geom_type[idx]

    if geom_type == mjtGeom.mjGEOM_PLANE:
      # Create a plane mesh
      return trimesh.creation.box((20, 20, 0.01))
    elif geom_type == mjtGeom.mjGEOM_SPHERE:
      radius = size[0]
      return trimesh.creation.icosphere(radius=radius, subdivisions=2)
    elif geom_type == mjtGeom.mjGEOM_BOX:
      dims = 2.0 * size
      return trimesh.creation.box(extents=dims)
    elif geom_type == mjtGeom.mjGEOM_MESH:
      mesh_id = mj_model.geom_dataid[idx]
      vert_start = mj_model.mesh_vertadr[mesh_id]
      vert_count = mj_model.mesh_vertnum[mesh_id]
      face_start = mj_model.mesh_faceadr[mesh_id]
      face_count = mj_model.mesh_facenum[mesh_id]

      verts = mj_model.mesh_vert[vert_start : (vert_start + vert_count), :]
      faces = mj_model.mesh_face[face_start : (face_start + face_count), :]

      mesh = trimesh.Trimesh(vertices=verts, faces=faces)
      mesh.fill_holes()
      mesh.fix_normals()
      return mesh

    elif geom_type == mjtGeom.mjGEOM_CAPSULE:
      r, half_len = size[0], size[1]
      return trimesh.creation.capsule(radius=r, height=2.0 * half_len)
    elif geom_type == mjtGeom.mjGEOM_CYLINDER:
      r, half_len = size[0], size[1]
      return trimesh.creation.cylinder(radius=r, height=2.0 * half_len)
    else:
      raise ValueError(f"Unsupported geom type {geom_type}")
