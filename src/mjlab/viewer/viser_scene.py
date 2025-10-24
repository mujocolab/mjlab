"""ViserMujocoScene manages all Viser visualization handles and state for MuJoCo models."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import trimesh
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj

from mjlab.viewer.viser_conversions import (
  get_body_name,
  is_fixed_body,
  merge_geoms,
  rotation_matrix_from_vectors,
)

try:
  import mujoco_warp as mjwarp
except ImportError:
  mjwarp = None  # type: ignore


@dataclass
class _Contact:
  """Contact data from MuJoCo."""

  pos: np.ndarray
  frame: np.ndarray  # 3x3 rotation matrix.
  force: np.ndarray  # Force in contact frame.
  dist: float
  included: bool


@dataclass
class _ContactPointVisual:
  """Visual representation data for a contact point."""

  position: np.ndarray
  orientation: np.ndarray  # Quaternion (wxyz).
  scale: np.ndarray  # [width, width, height].


@dataclass
class _ContactForceVisual:
  """Visual representation data for a contact force arrow."""

  shaft_position: np.ndarray
  shaft_orientation: np.ndarray  # Quaternion (wxyz).
  shaft_scale: np.ndarray  # [width, width, length].
  head_position: np.ndarray
  head_orientation: np.ndarray  # Quaternion (wxyz).
  head_scale: np.ndarray  # [width, width, width].


@dataclass
class ViserMujocoScene:
  """Manages Viser scene handles and visualization state for MuJoCo models."""

  # Core.
  server: viser.ViserServer
  mj_model: mujoco.MjModel
  mj_data: mujoco.MjData
  num_envs: int

  # Handles (created once).
  fixed_bodies_frame: viser.SceneNodeHandle = field(init=False)
  mesh_handles_by_group: dict[tuple[int, int], viser.BatchedGlbHandle] = field(
    default_factory=dict
  )
  contact_point_handle: viser.BatchedMeshHandle | None = None
  contact_force_shaft_handle: viser.BatchedMeshHandle | None = None
  contact_force_head_handle: viser.BatchedMeshHandle | None = None

  # Visualization settings (set directly or automatically updated by create_options_gui).
  current_env_idx: int = 0
  camera_tracking_enabled: bool = False
  show_only_selected: bool = False
  geom_groups_visible: list[bool] = field(
    default_factory=lambda: [True, True, True, False, False, False]
  )
  show_contact_points: bool = False
  show_contact_forces: bool = False
  contact_point_color: tuple[int, int, int] = (230, 153, 51)
  contact_force_color: tuple[int, int, int] = (255, 0, 0)
  meansize_override: float | None = None
  needs_update: bool = False
  _tracked_body_id: int | None = field(init=False, default=None)

  @staticmethod
  def create(
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    num_envs: int,
  ) -> ViserMujocoScene:
    """Create and populate scene with geometry.

    Visual geometry is created immediately. Collision geometry is created
    lazily when first needed.

    Args:
      server: Viser server instance.
      mj_model: MuJoCo model.
      num_envs: Number of parallel environments.

    Returns:
      ViserMujocoScene instance with scene populated.
    """
    mj_data = mujoco.MjData(mj_model)

    scene = ViserMujocoScene(
      server=server,
      mj_model=mj_model,
      mj_data=mj_data,
      num_envs=num_envs,
    )

    # Configure environment lighting.
    server.scene.configure_environment_map(environment_intensity=0.8)

    # Create frame for fixed world geometry.
    scene.fixed_bodies_frame = server.scene.add_frame("/fixed_bodies", show_axes=False)

    # Add fixed geometry (planes, terrain, etc.).
    scene._add_fixed_geometry()

    # Create mesh handles per geom group.
    scene._create_mesh_handles_by_group()

    # Find first non-fixed body for camera tracking.
    for body_id in range(mj_model.nbody):
      if not is_fixed_body(mj_model, body_id):
        scene._tracked_body_id = body_id
        break

    return scene

  def _is_collision_geom(self, geom_id: int) -> bool:
    """Check if a geom is a collision geom."""
    return (
      self.mj_model.geom_contype[geom_id] != 0
      or self.mj_model.geom_conaffinity[geom_id] != 0
    )

  def _sync_visibilities(self) -> None:
    """Synchronize all handle visibilities based on current flags."""
    # Geom group meshes.
    for (_body_id, group_id), handle in self.mesh_handles_by_group.items():
      handle.visible = group_id < 6 and self.geom_groups_visible[group_id]

    # Contact points.
    if self.contact_point_handle is not None and not self.show_contact_points:
      self.contact_point_handle.visible = False

    # Contact forces.
    if not self.show_contact_forces:
      if self.contact_force_shaft_handle is not None:
        self.contact_force_shaft_handle.visible = False
      if self.contact_force_head_handle is not None:
        self.contact_force_head_handle.visible = False

  def create_options_gui(self) -> None:
    """Add standard GUI controls that automatically update this scene's settings."""
    # Environment selection (only if multiple environments).
    if self.num_envs > 1:
      with self.server.gui.add_folder("Environment"):
        env_slider = self.server.gui.add_slider(
          "Select",
          min=0,
          max=self.num_envs - 1,
          step=1,
          initial_value=self.current_env_idx,
          hint=f"Select environment (0-{self.num_envs - 1})",
        )

        @env_slider.on_update
        def _(_) -> None:
          self.current_env_idx = int(env_slider.value)
          self.needs_update = True

        show_only_cb = self.server.gui.add_checkbox(
          "Hide others",
          initial_value=self.show_only_selected,
          hint="Show only the selected environment.",
        )

        @show_only_cb.on_update
        def _(_) -> None:
          self.show_only_selected = show_only_cb.value
          self.needs_update = True

    with self.server.gui.add_folder("Visualization"):
      slider_fov = self.server.gui.add_slider(
        "FOV (°)",
        min=20,
        max=150,
        step=1,
        initial_value=90,
        hint="Vertical FOV of viewer camera, in degrees.",
      )

      @slider_fov.on_update
      def _(_) -> None:
        for client in self.server.get_clients().values():
          client.camera.fov = np.radians(slider_fov.value)

      @self.server.on_client_connect
      def _(client: viser.ClientHandle) -> None:
        client.camera.fov = np.radians(slider_fov.value)

    # Geom group visibility controls.
    with self.server.gui.add_folder("Geom Groups"):
      for i in range(6):
        cb = self.server.gui.add_checkbox(
          f"Group {i}",
          initial_value=self.geom_groups_visible[i],
          hint=f"Show/hide geoms in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          self.geom_groups_visible[group_idx] = event.target.value
          self._sync_visibilities()
          self.needs_update = True

    # Contact visualization settings.
    with self.server.gui.add_folder("Contacts"):
      cb_contact_points = self.server.gui.add_checkbox(
        "Points",
        initial_value=False,
        hint="Toggle contact point visualization.",
      )
      contact_point_color = self.server.gui.add_rgb(
        "Points Color", initial_value=self.contact_point_color
      )
      cb_contact_forces = self.server.gui.add_checkbox(
        "Forces",
        initial_value=False,
        hint="Toggle contact force visualization.",
      )
      contact_force_color = self.server.gui.add_rgb(
        "Forces Color", initial_value=self.contact_force_color
      )
      meansize_input = self.server.gui.add_number(
        "Scale",
        step=self.mj_model.stat.meansize * 0.01,
        initial_value=self.mj_model.stat.meansize,
      )

      @cb_contact_points.on_update
      def _(_) -> None:
        self.show_contact_points = cb_contact_points.value
        self._sync_visibilities()
        self.needs_update = True

      @contact_point_color.on_update
      def _(_) -> None:
        self.contact_point_color = contact_point_color.value
        if self.contact_point_handle is not None:
          self.contact_point_handle.remove()
          self.contact_point_handle = None
        self.needs_update = True

      @cb_contact_forces.on_update
      def _(_) -> None:
        self.show_contact_forces = cb_contact_forces.value
        self._sync_visibilities()
        self.needs_update = True

      @contact_force_color.on_update
      def _(_) -> None:
        self.contact_force_color = contact_force_color.value
        if self.contact_force_shaft_handle is not None:
          self.contact_force_shaft_handle.remove()
          self.contact_force_shaft_handle = None
        if self.contact_force_head_handle is not None:
          self.contact_force_head_handle.remove()
          self.contact_force_head_handle = None
        self.needs_update = True

      @meansize_input.on_update
      def _(_) -> None:
        self.meansize_override = meansize_input.value
        self.needs_update = True

  def update(self, wp_data, env_idx: int | None = None) -> None:
    """Update scene from batched simulation data.

    Args:
      wp_data: Batched Warp simulation data (mjwarp.Data).
      env_idx: Environment index to visualize. If None, uses self.current_env_idx.
    """
    if env_idx is None:
      env_idx = self.current_env_idx

    body_xpos = wp_data.xpos.numpy()
    body_xmat = wp_data.xmat.numpy()
    mocap_pos = wp_data.mocap_pos.numpy()
    mocap_quat = wp_data.mocap_quat.numpy()
    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = body_xpos[env_idx, self._tracked_body_id, :].copy()
      scene_offset = -tracked_pos

    contacts = None
    if self.show_contact_points or self.show_contact_forces:
      self.mj_data.qpos[:] = wp_data.qpos.numpy()[env_idx]
      self.mj_data.qvel[:] = wp_data.qvel.numpy()[env_idx]
      self.mj_data.mocap_pos[:] = mocap_pos[env_idx]
      self.mj_data.mocap_quat[:] = mocap_quat[env_idx]
      mujoco.mj_forward(self.mj_model, self.mj_data)
      contacts = self._extract_contacts_from_mjdata(self.mj_data)

    self._update_visualization(
      body_xpos, body_xmat, mocap_pos, mocap_quat, env_idx, scene_offset, contacts
    )

  def update_from_mjdata(self, mj_data: mujoco.MjData) -> None:
    """Update scene from single-environment MuJoCo data.

    Args:
      mj_data: Single environment MuJoCo data.
    """
    body_xpos = mj_data.xpos[None, ...]
    body_xmat = mj_data.xmat.reshape(-1, 3, 3)[None, ...]
    mocap_pos = mj_data.mocap_pos[None, ...]
    mocap_quat = mj_data.mocap_quat[None, ...]
    env_idx = 0
    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = mj_data.xpos[self._tracked_body_id, :].copy()
      scene_offset = -tracked_pos

    contacts = None
    if self.show_contact_points or self.show_contact_forces:
      contacts = self._extract_contacts_from_mjdata(mj_data)

    self._update_visualization(
      body_xpos, body_xmat, mocap_pos, mocap_quat, env_idx, scene_offset, contacts
    )

  def _update_visualization(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    contacts: list[_Contact] | None,
  ) -> None:
    """Shared visualization update logic."""
    self.fixed_bodies_frame.position = scene_offset
    with self.server.atomic():
      body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz
      for (body_id, _group_id), handle in self.mesh_handles_by_group.items():
        if not handle.visible:
          continue
        # Check if this is a mocap body.
        mocap_id = self.mj_model.body_mocapid[body_id]
        if mocap_id >= 0:
          # Use mocap pos/quat for mocap bodies.
          if self.show_only_selected and self.num_envs > 1:
            single_pos = mocap_pos[env_idx, mocap_id, :] + scene_offset
            # Convert quaternion from xyzw to wxyz format.
            quat_xyzw = mocap_quat[env_idx, mocap_id, :]
            single_quat = np.array(
              [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            # Convert quaternion from xyzw to wxyz format for all envs.
            quat_xyzw = mocap_quat[:, mocap_id, :]
            quat_wxyz = np.stack(
              [quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]],
              axis=-1,
            )
            handle.batched_positions = mocap_pos[:, mocap_id, :] + scene_offset
            handle.batched_wxyzs = quat_wxyz
        else:
          # Use xpos/xmat for regular bodies.
          if self.show_only_selected and self.num_envs > 1:
            single_pos = body_xpos[env_idx, body_id, :] + scene_offset
            single_quat = body_xquat[env_idx, body_id, :]
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            handle.batched_positions = body_xpos[..., body_id, :] + scene_offset
            handle.batched_wxyzs = body_xquat[..., body_id, :]
      if contacts is not None:
        self._update_contact_visualization(contacts, scene_offset)

      self.server.flush()

  def _add_fixed_geometry(self) -> None:
    """Add fixed world geometry to the scene."""
    body_geoms_visual: dict[int, list[int]] = {}
    body_geoms_collision: dict[int, list[int]] = {}

    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]
      target = body_geoms_collision if self._is_collision_geom(i) else body_geoms_visual
      target.setdefault(body_id, []).append(i)

    # Process all bodies with geoms.
    all_bodies = set(body_geoms_visual.keys()) | set(body_geoms_collision.keys())

    for body_id in all_bodies:
      # Get body name.
      body_name = get_body_name(self.mj_model, body_id)

      # Fixed world geometry. We'll assume this is shared between all environments.
      if is_fixed_body(self.mj_model, body_id):
        # Create both visual and collision geoms for fixed bodies (terrain, floor, etc.)
        # but show them all since they're static.
        all_geoms = []
        if body_id in body_geoms_visual:
          all_geoms.extend(body_geoms_visual[body_id])
        if body_id in body_geoms_collision:
          all_geoms.extend(body_geoms_collision[body_id])

        if not all_geoms:
          continue

        # Iterate over geoms.
        nonplane_geom_ids: list[int] = []
        for geom_id in all_geoms:
          geom_type = self.mj_model.geom_type[geom_id]
          # Add plane geoms as infinite grids.
          if geom_type == mjtGeom.mjGEOM_PLANE:
            geom_name = mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, geom_id)
            self.server.scene.add_grid(
              f"/fixed_bodies/{body_name}/{geom_name}",
              # For infinite grids in viser 1.0.10, the width and height
              # parameters determined the region of the grid that can
              # receive shadows. We'll just make this really big for now.
              # In a future release of Viser these two args should ideally be
              # unnecessary.
              width=2000.0,
              height=2000.0,
              infinite_grid=True,
              fade_distance=50.0,
              shadow_opacity=0.2,
              position=self.mj_model.geom_pos[geom_id],
              wxyz=self.mj_model.geom_quat[geom_id],
            )
          else:
            nonplane_geom_ids.append(geom_id)

        # Handle non-plane geoms.
        if len(nonplane_geom_ids) > 0:
          self.server.scene.add_mesh_trimesh(
            f"/fixed_bodies/{body_name}",
            merge_geoms(self.mj_model, nonplane_geom_ids),
            cast_shadow=False,
            receive_shadow=0.2,
            position=self.mj_model.body(body_id).pos,
            wxyz=self.mj_model.body(body_id).quat,
            visible=True,
          )

  def _create_mesh_handles_by_group(self) -> None:
    """Create mesh handles for each geom group separately to allow independent toggling."""
    # Group geoms by (body_id, group_id).
    body_group_geoms: dict[tuple[int, int], list[int]] = {}

    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]

      # Skip fixed world geometry.
      if is_fixed_body(self.mj_model, body_id):
        continue

      geom_group = self.mj_model.geom_group[i]
      key = (body_id, geom_group)

      if key not in body_group_geoms:
        body_group_geoms[key] = []
      body_group_geoms[key].append(i)

    # Create handles for each (body, group) combination.
    with self.server.atomic():
      for (body_id, group_id), geom_indices in body_group_geoms.items():
        # Get body name.
        body_name = get_body_name(self.mj_model, body_id)

        # Merge geoms into a single mesh.
        mesh = merge_geoms(self.mj_model, geom_indices)
        lod_ratio = 1000.0 / mesh.vertices.shape[0]

        # Check if this group should be visible.
        visible = group_id < 6 and self.geom_groups_visible[group_id]

        # Create handle.
        handle = self.server.scene.add_batched_meshes_trimesh(
          f"/bodies/{body_name}/group{group_id}",
          mesh,
          batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
          visible=visible,
        )
        self.mesh_handles_by_group[(body_id, group_id)] = handle

  def _extract_contacts_from_mjdata(self, mj_data: mujoco.MjData) -> list[_Contact]:
    """Extract contact data from given MuJoCo data."""

    def make_contact(i: int) -> _Contact:
      con, force = mj_data.contact[i], np.zeros(6)
      mujoco.mj_contactForce(self.mj_model, mj_data, i, force)
      return _Contact(
        pos=con.pos.copy(),
        frame=con.frame.copy().reshape(3, 3),
        force=force[:3].copy(),
        dist=con.dist,
        included=con.efc_address >= 0,
      )

    return [make_contact(i) for i in range(mj_data.ncon)]

  def _update_contact_visualization(
    self, contacts: list[_Contact], scene_offset: np.ndarray
  ) -> None:
    """Update contact point and force visualization."""
    contact_points: list[_ContactPointVisual] = []
    contact_forces: list[_ContactForceVisual] = []

    meansize = self.meansize_override or self.mj_model.stat.meansize

    for contact in contacts:
      if not contact.included:
        continue

      # Transform force from contact frame to world frame.
      force_world = contact.frame.T @ contact.force
      force_mag = np.linalg.norm(force_world)

      # Contact point visualization (cylinder).
      if self.show_contact_points:
        contact_points.append(
          _ContactPointVisual(
            position=contact.pos + scene_offset,
            orientation=vtf.SO3.from_matrix(
              rotation_matrix_from_vectors(np.array([0, 0, 1]), contact.frame[0, :])
            ).wxyz,
            scale=np.array(
              [
                self.mj_model.vis.scale.contactwidth * meansize,
                self.mj_model.vis.scale.contactwidth * meansize,
                self.mj_model.vis.scale.contactheight * meansize,
              ]
            ),
          )
        )

      # Contact force visualization (arrow shaft + head).
      if self.show_contact_forces and force_mag > 1e-6:
        force_dir = force_world / force_mag
        arrow_length = (
          force_mag * (self.mj_model.vis.map.force / self.mj_model.stat.meanmass)
          if self.mj_model.stat.meanmass > 0
          else force_mag
        )
        arrow_width = self.mj_model.vis.scale.forcewidth * meansize
        force_quat = vtf.SO3.from_matrix(
          rotation_matrix_from_vectors(np.array([0, 0, 1]), force_dir)
        ).wxyz

        contact_forces.append(
          _ContactForceVisual(
            shaft_position=contact.pos + scene_offset,
            shaft_orientation=force_quat,
            shaft_scale=np.array([arrow_width, arrow_width, arrow_length]),
            head_position=contact.pos + scene_offset + force_dir * arrow_length,
            head_orientation=force_quat,
            head_scale=np.array([arrow_width, arrow_width, arrow_width]),
          )
        )

    # Update or create contact point handle.
    if contact_points:
      positions = np.array([p.position for p in contact_points], dtype=np.float32)
      orientations = np.array([p.orientation for p in contact_points], dtype=np.float32)
      scales = np.array([p.scale for p in contact_points], dtype=np.float32)
      if self.contact_point_handle is None:
        mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
        self.contact_point_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/points",
          mesh.vertices,
          mesh.faces,
          batched_wxyzs=orientations,
          batched_positions=positions,
          batched_scales=scales,
          batched_colors=np.array(self.contact_point_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
      self.contact_point_handle.batched_positions = positions
      self.contact_point_handle.batched_wxyzs = orientations
      self.contact_point_handle.batched_scales = scales
      self.contact_point_handle.visible = True
    elif self.contact_point_handle is not None:
      self.contact_point_handle.visible = False

    # Update or create contact force handles (shaft and head separately).
    if contact_forces:
      shaft_positions = np.array(
        [f.shaft_position for f in contact_forces], dtype=np.float32
      )
      shaft_orientations = np.array(
        [f.shaft_orientation for f in contact_forces], dtype=np.float32
      )
      shaft_scales = np.array([f.shaft_scale for f in contact_forces], dtype=np.float32)
      head_positions = np.array(
        [f.head_position for f in contact_forces], dtype=np.float32
      )
      head_orientations = np.array(
        [f.head_orientation for f in contact_forces], dtype=np.float32
      )
      head_scales = np.array([f.head_scale for f in contact_forces], dtype=np.float32)
      if self.contact_force_shaft_handle is None:
        shaft_mesh = trimesh.creation.cylinder(radius=0.4, height=1.0)
        shaft_mesh.apply_translation([0, 0, 0.5])
        self.contact_force_shaft_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/forces/shaft",
          shaft_mesh.vertices,
          shaft_mesh.faces,
          batched_wxyzs=shaft_orientations,
          batched_positions=shaft_positions,
          batched_scales=shaft_scales,
          batched_colors=np.array(self.contact_force_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
        head_mesh = trimesh.creation.cone(radius=1.0, height=1.0, sections=8)
        self.contact_force_head_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/forces/head",
          head_mesh.vertices,
          head_mesh.faces,
          batched_wxyzs=head_orientations,
          batched_positions=head_positions,
          batched_scales=head_scales,
          batched_colors=np.array(self.contact_force_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
      assert self.contact_force_shaft_handle is not None
      assert self.contact_force_head_handle is not None
      self.contact_force_shaft_handle.batched_positions = shaft_positions
      self.contact_force_shaft_handle.batched_wxyzs = shaft_orientations
      self.contact_force_shaft_handle.batched_scales = shaft_scales
      self.contact_force_shaft_handle.visible = True
      self.contact_force_head_handle.batched_positions = head_positions
      self.contact_force_head_handle.batched_wxyzs = head_orientations
      self.contact_force_head_handle.batched_scales = head_scales
      self.contact_force_head_handle.visible = True
    elif (
      self.contact_force_shaft_handle is not None
      and self.contact_force_head_handle is not None
    ):
      self.contact_force_shaft_handle.visible = (
        self.contact_force_head_handle.visible
      ) = False
