"""Mjlab viewer based on Viser.

Adapted from an MJX visualizer by Chung Min Kim: https://github.com/chungmin99/
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import mujoco
import numpy as np
import trimesh
import trimesh.visual
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj  # type: ignore
from typing_extensions import override

from mjlab.sim.sim import Simulation
from mjlab.viewer.base import BaseViewer, EnvProtocol, PolicyProtocol, VerbosityLevel
from mjlab.viewer.viser_conversions import mujoco_mesh_to_trimesh
from mjlab.viewer.viser_reward_plotter import ViserRewardPlotter


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
    self._reward_plotter: Optional[ViserRewardPlotter] = None

  @override
  def setup(self) -> None:
    """Setup the viewer resources."""

    self._server = viser.ViserServer(label="mjlab")
    # Separate handle storage for visual and collision meshes
    self._mesh_visual_handles: dict[int, viser.BatchedGlbHandle] | None = None
    self._mesh_collision_handles: dict[int, viser.BatchedGlbHandle] | None = None
    # Contact visualization handles
    self._contact_point_handle: viser.BatchedGlbHandle | None = None
    self._contact_force_shaft_handle: viser.BatchedGlbHandle | None = None
    self._contact_force_head_handle: viser.BatchedGlbHandle | None = None
    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._batch_size = self.env.num_envs
    self._show_contact_points = False
    self._show_contact_forces = False
    self._meansize_override: float | None = None
    self._camera_tracking = False
    self._camera_distance = 3.0
    self._camera_azimuth = 90.0  # degrees
    self._camera_elevation = -20.0  # degrees
    self._contact_point_color = (230, 153, 51)  # RGB 0-255, MuJoCo default
    self._contact_force_color = (179, 230, 230)  # RGB 0-255, MuJoCo default

    self._counter = 0
    self._env_idx = 0
    self._show_only_selected_env = False

    # Set up lighting.
    self._server.scene.configure_environment_map(environment_intensity=0.8)

    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    mj_model = sim.mj_model

    # Create tabs
    tabs = self._server.gui.add_tab_group()

    # Main tab with simulation controls and display settings
    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
      # Status display
      with self._server.gui.add_folder("Info"):
        self._status_html = self._server.gui.add_html("")

      # Simulation controls
      with self._server.gui.add_folder("Simulation"):
        # Play/Pause button
        self._pause_button = self._server.gui.add_button(
          "Play" if self._is_paused else "Pause",
          icon=viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE,
        )

        @self._pause_button.on_click
        def _(_) -> None:
          self.toggle_pause()
          self._pause_button.label = "Play" if self._is_paused else "Pause"
          self._pause_button.icon = (
            viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE
          )
          self._update_status_display()

        # Reset button
        reset_button = self._server.gui.add_button("Reset Environment")

        @reset_button.on_click
        def _(_) -> None:
          self.reset_environment()
          self._update_status_display()

        # Speed controls
        speed_buttons = self._server.gui.add_button_group(
          "Speed",
          options=["Slower", "Faster"],
        )

        @speed_buttons.on_click
        def _(event) -> None:
          if event.target.value == "Slower":
            self.decrease_speed()
          else:
            self.increase_speed()
          self._update_status_display()

      # Visualization settings
      with self._server.gui.add_folder("Visualization"):
        cb_visual = self._server.gui.add_checkbox("Visual geom", initial_value=True)
        cb_collision = self._server.gui.add_checkbox(
          "Collision geom", initial_value=False
        )
        slider_fov = self._server.gui.add_slider(
          "FOV (°)",
          min=20,
          max=150,
          step=1,
          initial_value=90,
          hint="Vertical FOV of viewer camera, in degrees.",
        )

        @cb_visual.on_update
        def _(_) -> None:
          if cb_visual.value:
            self._ensure_visual_handles_exist()

          if self._mesh_visual_handles is not None:
            for handle in self._mesh_visual_handles.values():
              handle.visible = cb_visual.value

        @cb_collision.on_update
        def _(_) -> None:
          visibility = cb_collision.value
          if visibility:
            self._ensure_collision_handles_exist()

          if self._mesh_collision_handles is not None:
            for handle in self._mesh_collision_handles.values():
              # If hiding meshes: throw them off the screen, because when
              # they're shown again the current positions will be outdated.
              if not visibility:
                handle.batched_positions = handle.batched_positions - 2000.0
              handle.visible = visibility

        # Update FOV when a new client connects.
        @slider_fov.on_update
        def _(_) -> None:
          for client in self._server.get_clients().values():
            client.camera.fov = np.radians(slider_fov.value)

        # Set initial FOV when clients connect.
        @self._server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
          client.camera.fov = np.radians(slider_fov.value)

      # Environment selection if multiple environments
      if self.env.num_envs > 1:
        with self._server.gui.add_folder("Environment"):
          # Navigation buttons
          env_nav_buttons = self._server.gui.add_button_group(
            "Navigate",
            options=["Previous", "Next"],
          )

          @env_nav_buttons.on_click
          def _(event) -> None:
            # Just update the slider, which will trigger its callback
            if event.target.value == "Previous":
              new_idx = (self._env_idx - 1) % self.env.num_envs
            else:
              new_idx = (self._env_idx + 1) % self.env.num_envs
            self._env_slider.value = new_idx

          # Environment slider for direct selection
          self._env_slider = self._server.gui.add_slider(
            "Select",
            min=0,
            max=self.env.num_envs - 1,
            step=1,
            initial_value=0,
          )

          @self._env_slider.on_update
          def _(_) -> None:
            self._env_idx = int(self._env_slider.value)
            self._update_status_display()
            if self._reward_plotter:
              self._reward_plotter.clear_histories()

          # Checkbox to show only selected environment
          self._show_only_selected_cb = self._server.gui.add_checkbox(
            "Show only this environment", initial_value=False
          )

          @self._show_only_selected_cb.on_update
          def _(_) -> None:
            self._show_only_selected_env = self._show_only_selected_cb.value

          # Contact visualization settings
          with self._server.gui.add_folder("Contacts"):
            cb_contact_points = self._server.gui.add_checkbox(
              "Points",
              initial_value=False,
              hint="Toggle contact point visualization for the selected environment.",
            )
            contact_point_color = self._server.gui.add_rgb(
              "Points Color", initial_value=self._contact_point_color
            )
            cb_contact_forces = self._server.gui.add_checkbox(
              "Forces",
              initial_value=False,
              hint="Toggle contact force visualization for the selected environment.",
            )
            contact_force_color = self._server.gui.add_rgb(
              "Forces Color", initial_value=self._contact_force_color
            )
            meansize_input = self._server.gui.add_number(
              "Scale",
              min=0.001,
              max=1.0,
              step=0.001,
              initial_value=mj_model.stat.meansize,
            )

            @cb_contact_points.on_update
            def _(_) -> None:
              self._show_contact_points = cb_contact_points.value
              # Immediately hide if disabled
              if not cb_contact_points.value and self._contact_point_handle is not None:
                self._contact_point_handle.visible = False

            @contact_point_color.on_update
            def _(_) -> None:
              self._contact_point_color = contact_point_color.value
              # Force recreation of contact point handle with new color
              if self._contact_point_handle is not None:
                self._contact_point_handle.remove()
                self._contact_point_handle = None

            @cb_contact_forces.on_update
            def _(_) -> None:
              self._show_contact_forces = cb_contact_forces.value
              # Immediately hide if disabled
              if not cb_contact_forces.value:
                if self._contact_force_shaft_handle is not None:
                  self._contact_force_shaft_handle.visible = False
                if self._contact_force_head_handle is not None:
                  self._contact_force_head_handle.visible = False

            @contact_force_color.on_update
            def _(_) -> None:
              self._contact_force_color = contact_force_color.value
              # Force recreation of contact force handles with new color
              if self._contact_force_shaft_handle is not None:
                self._contact_force_shaft_handle.remove()
                self._contact_force_shaft_handle = None
              if self._contact_force_head_handle is not None:
                self._contact_force_head_handle.remove()
                self._contact_force_head_handle = None

            @meansize_input.on_update
            def _(_) -> None:
              self._meansize_override = meansize_input.value

          # Camera tracking controls
          with self._server.gui.add_folder("Camera"):
            cb_camera_tracking = self._server.gui.add_checkbox(
              "Enable tracking", initial_value=False
            )
            camera_distance_slider = self._server.gui.add_slider(
              "Distance",
              min=0.5,
              max=10.0,
              step=0.1,
              initial_value=3.0,
            )
            camera_azimuth_slider = self._server.gui.add_slider(
              "Azimuth",
              min=-180.0,
              max=180.0,
              step=5.0,
              initial_value=90.0,
            )
            camera_elevation_slider = self._server.gui.add_slider(
              "Elevation",
              min=-89.0,
              max=89.0,
              step=5.0,
              initial_value=-20.0,
            )

            @cb_camera_tracking.on_update
            def _(_) -> None:
              self._camera_tracking = cb_camera_tracking.value

            @camera_distance_slider.on_update
            def _(_) -> None:
              self._camera_distance = camera_distance_slider.value

            @camera_azimuth_slider.on_update
            def _(_) -> None:
              self._camera_azimuth = camera_azimuth_slider.value

            @camera_elevation_slider.on_update
            def _(_) -> None:
              self._camera_elevation = camera_elevation_slider.value

    # Reward plots tab
    if hasattr(self.env.unwrapped, "reward_manager"):
      with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
        # Get reward term names and create reward plotter
        term_names = [
          name
          for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self._env_idx
          )
        ]
        self._reward_plotter = ViserRewardPlotter(self._server, term_names)

    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)

    # Group geoms by their parent body and type (visual/collision)
    body_geoms_visual: dict[int, list[int]] = {}
    body_geoms_collision: dict[int, list[int]] = {}

    for i in range(mj_model.ngeom):
      body_id = mj_model.geom_bodyid[i]
      # Check if it's a collision geom
      is_collision = mj_model.geom_contype[i] != 0 or mj_model.geom_conaffinity[i] != 0

      if is_collision:
        if body_id not in body_geoms_collision:
          body_geoms_collision[body_id] = []
        body_geoms_collision[body_id].append(i)
      else:
        if body_id not in body_geoms_visual:
          body_geoms_visual[body_id] = []
        body_geoms_visual[body_id].append(i)

    # Process visual and collision geoms separately for each body
    all_bodies = set(body_geoms_visual.keys()) | set(body_geoms_collision.keys())

    for body_id in all_bodies:
      # Get body name
      body_name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
      if not body_name:
        body_name = f"body_{body_id}"

      # Fixed world geometry. We'll assume this is shared between all
      # environments.
      if mj_model.body_dofnum[body_id] == 0 and mj_model.body_parentid[body_id] == 0:
        for body_geoms_dict, visual_or_collision in [
          (body_geoms_visual, "visual"),
          (body_geoms_collision, "collision"),
        ]:
          if body_id not in body_geoms_dict:
            continue

          # Iterate over geoms.
          nonplane_geom_ids: list[int] = []
          for geom_id in body_geoms_dict[body_id]:
            geom_type = mj_model.geom_type[geom_id]
            # Add plane geoms as infinite grids.
            if geom_type == mjtGeom.mjGEOM_PLANE:
              geom_id = body_geoms_dict[body_id][0]
              geom_type = mj_model.geom_type[geom_id]
              if geom_type == mjtGeom.mjGEOM_PLANE:
                geom_name = mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, geom_id)
                self._server.scene.add_grid(
                  f"/fixed_bodies/{body_name}/{geom_name}/{visual_or_collision}",
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
                  position=mj_model.geom_pos[geom_id],
                  wxyz=mj_model.geom_quat[geom_id],
                )
                continue
            else:
              nonplane_geom_ids.append(geom_id)

          # Handle non-plane geoms later.
          if len(nonplane_geom_ids) > 0:
            # geom is visible if it is a terrain or a visual geom
            visible = (body_name == "terrain") or (visual_or_collision == "visual")
            self._server.scene.add_mesh_trimesh(
              f"/fixed_bodies/{body_name}/{visual_or_collision}",
              self._merge_geoms(mj_model, nonplane_geom_ids),
              cast_shadow=False,
              receive_shadow=0.2,
              position=mj_model.body(body_id).pos,
              wxyz=mj_model.body(body_id).quat,
              visible=visible,
            )

    # Create visual handles by default on startup
    self._ensure_visual_handles_exist()

  def _merge_geoms(self, mj_model, geom_indices: list[int]) -> trimesh.Trimesh:
    """Merge multiple geoms into a single trimesh."""
    meshes_to_concat = []
    for geom_id in geom_indices:
      geom_name = mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, geom_id)
      if not geom_name:
        geom_name = f"geom_{geom_id}"

      # Get geom type
      geom_type = mj_model.geom_type[geom_id]

      # Create or get mesh for this geom
      if geom_type == mjtGeom.mjGEOM_MESH:
        mesh = mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
      else:
        mesh = self._create_mesh(mj_model, geom_id)

      # Transform mesh to geom's local pose relative to body
      pos = mj_model.geom_pos[geom_id]
      quat = mj_model.geom_quat[geom_id]  # (w, x, y, z)

      # Apply transformation to mesh
      transform = np.eye(4)
      transform[:3, :3] = vtf.SO3(quat).as_matrix()
      transform[:3, 3] = pos
      mesh.apply_transform(transform)

      meshes_to_concat.append(mesh)

    if len(meshes_to_concat) == 1:
      combined_mesh = meshes_to_concat[0]
    else:
      combined_mesh = trimesh.util.concatenate(meshes_to_concat)
    return combined_mesh

  def _create_mesh_handles(
    self, mesh_type: Literal["visual", "collision"], visible: bool
  ) -> dict[int, viser.BatchedGlbHandle]:
    """Create mesh handles for either visual or collision geometry.

    Args:
      mesh_type: Either "visual" or "collision"
      visible: Whether the meshes should be initially visible

    Returns:
      Dictionary mapping body_id to handles
    """
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    mj_model = sim.mj_model

    # Group geoms by body
    body_geoms: dict[int, list[int]] = {}

    for i in range(mj_model.ngeom):
      body_id = mj_model.geom_bodyid[i]
      is_collision = mj_model.geom_contype[i] != 0 or mj_model.geom_conaffinity[i] != 0

      # Add geom to body's list if it matches the type we're looking for
      if (mesh_type == "collision" and is_collision) or (
        mesh_type == "visual" and not is_collision
      ):
        if body_id not in body_geoms:
          body_geoms[body_id] = []
        body_geoms[body_id].append(i)

    handles = {}
    with self._server.atomic():
      for body_id, geom_indices in body_geoms.items():
        # Skip fixed world geometry
        if mj_model.body_dofnum[body_id] == 0 and mj_model.body_parentid[body_id] == 0:
          continue

        # Get body name
        body_name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
          body_name = f"body_{body_id}"

        # Merge geoms into a single mesh
        mesh = self._merge_geoms(mj_model, geom_indices)
        lod_ratio = 1000.0 / mesh.vertices.shape[0]

        # Create handle
        handle = self._server.scene.add_batched_meshes_trimesh(
          f"/bodies/{body_name}/{mesh_type}",
          mesh,
          batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
            self._batch_size, axis=0
          ),
          batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
            self._batch_size, axis=0
          ),
          lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
          visible=visible,
        )
        handles[body_id] = handle

    return handles

  def _ensure_visual_handles_exist(self) -> None:
    """Create visual mesh handles if they don't exist yet."""
    if self._mesh_visual_handles is not None:
      return  # Already created

    self._mesh_visual_handles = self._create_mesh_handles("visual", visible=True)

  def _ensure_collision_handles_exist(self) -> None:
    """Create collision mesh handles if they don't exist yet."""
    if self._mesh_collision_handles is not None:
      return  # Already created

    self._mesh_collision_handles = self._create_mesh_handles("collision", visible=True)

  @override
  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""

    # Update counter
    self._counter += 1

    # Update status display and reward plots less frequently.
    if self._counter % 10 == 0:
      self._update_status_display()
      if self._reward_plotter is not None and not self._is_paused:
        terms = list(
          self.env.unwrapped.reward_manager.get_active_iterable_terms(self._env_idx)
        )
        self._reward_plotter.update(terms)

    # The rest of this method is environment state syncing.
    # It's fine to do this every other policy step to reduce bandwidth usage.
    if self._counter % 2 != 0:
      return

    # Update camera tracking and contacts (after early return so it syncs with mesh updates)
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    wp_data = sim.wp_data
    mj_model = sim.mj_model

    # Check if we need to run mj_forward for camera tracking or contact visualization
    needs_mj_forward = (
      self._camera_tracking or self._show_contact_points or self._show_contact_forces
    )

    if needs_mj_forward:
      # Copy qpos/qvel to mj_data and run forward once (shared for tracking and contacts)
      mj_data = sim.mj_data
      mj_data.qpos[:] = wp_data.qpos.numpy()[self._env_idx]
      mj_data.qvel[:] = wp_data.qvel.numpy()[self._env_idx]
      mujoco.mj_forward(mj_model, mj_data)

    # We'll make a copy of the relevant state, then do the update itself asynchronously.
    geom_xpos = wp_data.geom_xpos.numpy()
    assert geom_xpos.shape == (self._batch_size, mj_model.ngeom, 3)
    geom_xmat = wp_data.geom_xmat.numpy()
    assert geom_xmat.shape == (self._batch_size, mj_model.ngeom, 3, 3)

    # Get body positions and orientations from warp data
    body_xpos = wp_data.xpos.numpy()  # Shape: (batch_size, nbody, 3)
    body_xmat = wp_data.xmat.numpy()  # Shape: (batch_size, nbody, 3, 3)

    # Get contact data if contact visualization is enabled
    # Only visualize contacts for the selected environment to reduce load
    contact_data = None
    env_origin = None
    if self._show_contact_points or self._show_contact_forces:
      # Extract contact data from already-computed mj_data (no need to call mj_forward again)
      contact_data = self._extract_contact_data(sim.mj_model, sim.mj_data)
      # Get world body (body 0) position as environment origin
      env_origin = body_xpos[self._env_idx, 0, :]  # Shape: (3,)

      # Get the body to track from viewer config
      # Use subtree center of mass for smoother tracking (like MuJoCo does)
    if self._camera_tracking:
      if self.cfg and self.cfg.asset_name and self.cfg.body_name:
        # Use the configured body for tracking
        from mjlab.entity.entity import Entity

        robot: Entity = self.env.unwrapped.scene[self.cfg.asset_name]
        if self.cfg.body_name not in robot.body_names:
          raise ValueError(
            f"Body '{self.cfg.body_name}' not found in asset '{self.cfg.asset_name}'"
          )
        body_id_list, _ = robot.find_bodies(self.cfg.body_name)
        root_body_id = robot.indexing.bodies[body_id_list[0]].id
      else:
        # Fallback: use body 1 (first body after world)
        root_body_id = 1
      camera_lookat = sim.mj_data.subtree_com[root_body_id].copy()
    else:
      camera_lookat = None

    def update_mujoco() -> None:
      with self._server.atomic():
        body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz

        # Update both visual and collision handles symmetrically
        for handles_dict in [self._mesh_visual_handles, self._mesh_collision_handles]:
          if handles_dict is None:
            continue  # Handles not created yet

          for body_id, handle in handles_dict.items():
            # Skip if handle is not visible
            if not handle.visible:
              continue

            # Update position and orientation for this body
            if self._show_only_selected_env and self.env.num_envs > 1:
              # Show only the selected environment at the origin (0,0,0)
              single_pos = body_xpos[self._env_idx, body_id, :]
              single_quat = body_xquat[
                self._env_idx, body_id, :
              ]  # Replicate single environment data for all batch slots
              handle.batched_positions = np.tile(
                single_pos[None, :], (self._batch_size, 1)
              )
              handle.batched_wxyzs = np.tile(
                single_quat[None, :], (self._batch_size, 1)
              )
            else:
              # Show all environments with offsets
              handle.batched_positions = body_xpos[..., body_id, :]
              handle.batched_wxyzs = body_xquat[..., body_id, :]

        # Update contact visualization
        if contact_data is not None and env_origin is not None:
          self._update_contact_visualization(contact_data, env_origin)

        # Synchronize camera tracking if enabled.
        if camera_lookat is not None:
          self._update_camera_tracking(camera_lookat)

        self._server.flush()

    self._threadpool.submit(update_mujoco)

  @override
  def sync_viewer_to_env(self) -> None:
    """Synchronize viewer state to environment (e.g., perturbations)."""
    # Does nothing for Viser.
    pass

  def reset_environment(self) -> None:
    """Extend BaseViewer.reset_environment to clear reward histories."""
    super().reset_environment()
    if self._reward_plotter:
      self._reward_plotter.clear_histories()

  @override
  def close(self) -> None:
    """Close the viewer and cleanup resources."""
    if self._reward_plotter:
      self._reward_plotter.cleanup()
    self._threadpool.shutdown(wait=True)
    self._server.stop()

  @override
  def is_running(self) -> bool:
    """Check if viewer is running."""
    return True  # Viser runs until process is killed

  def _update_status_display(self) -> None:
    """Update the HTML status display."""
    self._status_html.content = f"""
      <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
        <strong>Status:</strong> {"Paused" if self._is_paused else "Running"}<br/>
        <strong>Steps:</strong> {self._step_count}<br/>
        <strong>Speed:</strong> {self._time_multiplier:.0%}
      </div>
      """

  def _extract_contact_data(self, mj_model, mj_data) -> dict:
    """Extract contact data from already-computed mj_data.

    Args:
      mj_model: MuJoCo model
      mj_data: MuJoCo data (must have mj_forward already called)

    Returns:
      Dictionary with contact information
    """
    # Extract contact information
    contacts = []
    for i in range(mj_data.ncon):
      con = mj_data.contact[i]
      # Get contact force
      force = np.zeros(6)
      mujoco.mj_contactForce(mj_model, mj_data, i, force)

      contacts.append(
        {
          "pos": con.pos.copy(),
          "frame": con.frame.copy().reshape(3, 3),
          "force": force[:3].copy(),  # Only normal + friction forces
          "dist": con.dist,
          "included": con.efc_address >= 0,
        }
      )

    return {"contacts": contacts}

  def _update_contact_visualization(
    self, contact_data: dict, env_origin: np.ndarray
  ) -> None:
    """Update contact point and force visualization.

    Args:
      contact_data: Contact data dict for the selected environment
      env_origin: Array of shape (3,) with environment origin
    """
    # Collect all contact points and forces for the selected environment
    all_positions = []
    all_orientations = []
    all_scales = []
    force_shaft_positions = []
    force_shaft_orientations = []
    force_shaft_scales = []
    force_head_positions = []
    force_head_orientations = []
    force_head_scales = []

    contacts = contact_data["contacts"]

    for contact in contacts:
      if not contact["included"]:
        continue

      pos = contact["pos"]
      frame = contact["frame"]  # 3x3 rotation matrix (rows are basis vectors)
      force_contact_frame = contact["force"]  # Force in contact frame

      # Transform force from contact frame to world frame
      # Frame rows are [normal, tangent1, tangent2], so transpose to convert
      force_world = frame.T @ force_contact_frame
      force_mag = np.linalg.norm(force_world)

      # Contact point visualization (cylinder)
      if self._show_contact_points:
        # Add environment origin offset
        display_pos = pos + env_origin
        all_positions.append(display_pos)
        # Contact frame: first row is normal, need to align cylinder z-axis with normal
        normal = frame[0, :]  # Contact normal (first row of contact frame)
        contact_rot = self._rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)
        quat = vtf.SO3.from_matrix(contact_rot).wxyz
        all_orientations.append(quat)
        # Use MuJoCo's contact visualization scale
        sim = self.env.unwrapped.sim
        assert isinstance(sim, Simulation)
        meansize = self._get_meansize()
        contact_width = sim.mj_model.vis.scale.contactwidth * meansize
        contact_height = sim.mj_model.vis.scale.contactheight * meansize
        all_scales.append([contact_width, contact_width, contact_height])

      # Contact force visualization (arrow shaft + head)
      if self._show_contact_forces and force_mag > 1e-6:
        # Add environment origin offset
        force_base_pos = pos + env_origin

        # Compute arrow orientation (arrow points in direction of force)
        force_dir = force_world / force_mag
        force_rot = self._rotation_matrix_from_vectors(np.array([0, 0, 1]), force_dir)
        force_quat = vtf.SO3.from_matrix(force_rot).wxyz

        # Scale arrow by force magnitude
        sim = self.env.unwrapped.sim
        assert isinstance(sim, Simulation)
        meansize = self._get_meansize()
        # Use MuJoCo's force scaling: mju_scl3(vec, vec, m->vis.map.force/m->stat.meanmass)
        # This scales the force vector by (map.force / meanmass) - that's the arrow length
        force_scale = sim.mj_model.vis.map.force
        mean_mass = sim.mj_model.stat.meanmass
        if mean_mass > 0:
          arrow_length = force_mag * (force_scale / mean_mass)
        else:
          arrow_length = force_mag
        arrow_width = sim.mj_model.vis.scale.forcewidth * meansize

        # Shaft: stretches in z-direction
        force_shaft_positions.append(force_base_pos)
        force_shaft_orientations.append(force_quat)
        force_shaft_scales.append([arrow_width, arrow_width, arrow_length])

        # Head: fixed size, positioned at the tip of the shaft
        # Tip is at arrow_length in the force direction from base
        head_pos = force_base_pos + force_dir * arrow_length
        force_head_positions.append(head_pos)
        force_head_orientations.append(force_quat)
        # Head has fixed size based on arrow_width
        force_head_scales.append([arrow_width, arrow_width, arrow_width])

    # Update or create contact point handle
    if self._show_contact_points and len(all_positions) > 0:
      positions_arr = np.array(all_positions)
      orientations_arr = np.array(all_orientations)
      scales_arr = np.array(all_scales)

      if self._contact_point_handle is None:
        # Create cylinder mesh for contact points
        cylinder_mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
        # Convert RGB 0-255 to 0-1 range
        color_01 = [c / 255.0 for c in self._contact_point_color]
        cylinder_mesh.visual = trimesh.visual.TextureVisuals(
          material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color_01 + [1.0],  # Add alpha channel
          )
        )
        self._contact_point_handle = self._server.scene.add_batched_meshes_trimesh(
          "/contacts/points",
          cylinder_mesh,
          batched_wxyzs=orientations_arr,
          batched_positions=positions_arr,
          batched_scales=scales_arr,
          lod="off",
          visible=True,
          cast_shadow=False,
        )
      else:
        self._contact_point_handle.batched_positions = positions_arr
        self._contact_point_handle.batched_wxyzs = orientations_arr
        self._contact_point_handle.batched_scales = scales_arr
        self._contact_point_handle.visible = True
    elif self._contact_point_handle is not None:
      self._contact_point_handle.visible = False

    # Update or create contact force handles (shaft and head separately)
    if self._show_contact_forces and len(force_shaft_positions) > 0:
      shaft_positions_arr = np.array(force_shaft_positions)
      shaft_orientations_arr = np.array(force_shaft_orientations)
      shaft_scales_arr = np.array(force_shaft_scales)
      head_positions_arr = np.array(force_head_positions)
      head_orientations_arr = np.array(force_head_orientations)
      head_scales_arr = np.array(force_head_scales)

      if self._contact_force_shaft_handle is None:
        # Create shaft mesh (cylinder) - unit height, will be stretched
        shaft_mesh = trimesh.creation.cylinder(radius=0.4, height=1.0)
        shaft_mesh.apply_translation([0, 0, 0.5])  # Center at z=0.5
        # Convert RGB 0-255 to 0-1 range
        color_01 = [c / 255.0 for c in self._contact_force_color]
        shaft_mesh.visual = trimesh.visual.TextureVisuals(
          material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color_01 + [1.0],  # Add alpha channel
          )
        )
        self._contact_force_shaft_handle = (
          self._server.scene.add_batched_meshes_trimesh(
            "/contacts/forces/shaft",
            shaft_mesh,
            batched_wxyzs=shaft_orientations_arr,
            batched_positions=shaft_positions_arr,
            batched_scales=shaft_scales_arr,
            lod="off",
            visible=True,
            cast_shadow=False,
          )
        )

        # Create head mesh (cone) - fixed size
        head_mesh = trimesh.creation.cone(radius=1.0, height=1.5, sections=8)
        head_mesh.apply_translation([0, 0, 0.75])  # Center at z=0.75
        # Convert RGB 0-255 to 0-1 range (reuse same color as shaft)
        head_mesh.visual = trimesh.visual.TextureVisuals(
          material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color_01 + [1.0],  # Add alpha channel
          )
        )
        self._contact_force_head_handle = self._server.scene.add_batched_meshes_trimesh(
          "/contacts/forces/head",
          head_mesh,
          batched_wxyzs=head_orientations_arr,
          batched_positions=head_positions_arr,
          batched_scales=head_scales_arr,
          lod="off",
          visible=True,
          cast_shadow=False,
        )
      else:
        assert self._contact_force_head_handle is not None
        self._contact_force_shaft_handle.batched_positions = shaft_positions_arr
        self._contact_force_shaft_handle.batched_wxyzs = shaft_orientations_arr
        self._contact_force_shaft_handle.batched_scales = shaft_scales_arr
        self._contact_force_shaft_handle.visible = True

        self._contact_force_head_handle.batched_positions = head_positions_arr
        self._contact_force_head_handle.batched_wxyzs = head_orientations_arr
        self._contact_force_head_handle.batched_scales = head_scales_arr
        self._contact_force_head_handle.visible = True
    elif self._contact_force_shaft_handle is not None:
      assert self._contact_force_head_handle is not None
      self._contact_force_shaft_handle.visible = False
      self._contact_force_head_handle.visible = False

  def _update_camera_tracking(self, lookat: np.ndarray) -> None:
    """Update camera position to track the specified lookat point.

    Args:
      lookat: 3D point to look at (center of mass)
    """
    # Convert angles to radians
    azimuth_rad = np.deg2rad(self._camera_azimuth)
    elevation_rad = np.deg2rad(self._camera_elevation)

    # Calculate forward vector from azimuth and elevation
    # This matches MuJoCo's camera frame calculation
    forward = np.array(
      [
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad),
      ]
    )

    # Camera position is lookat - forward * distance
    camera_pos = lookat - forward * self._camera_distance

    # Update all connected clients
    for client in self._server.get_clients().values():
      with client.atomic():
        client.camera.position = camera_pos
        client.camera.look_at = lookat

  def _get_meansize(self) -> float:
    """Get the meansize value, using override if set."""
    if self._meansize_override is not None:
      return self._meansize_override
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    return sim.mj_model.stat.meansize

  @staticmethod
  def _rotation_matrix_from_vectors(
    from_vec: np.ndarray, to_vec: np.ndarray
  ) -> np.ndarray:
    """Create rotation matrix that rotates from_vec to to_vec using Rodrigues formula."""
    if np.allclose(from_vec, to_vec):
      return np.eye(3)
    if np.allclose(from_vec, -to_vec):
      return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    # Rodrigues rotation formula
    v = np.cross(from_vec, to_vec)
    s = np.linalg.norm(v)
    c = np.dot(from_vec, to_vec)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

  @staticmethod
  def _create_mesh(mj_model, idx: int) -> trimesh.Trimesh:
    """
    Create a trimesh object from a geom in the MuJoCo model.
    """
    size = mj_model.geom_size[idx]
    geom_type = mj_model.geom_type[idx]

    # Get geom RGBA color if available
    rgba = mj_model.geom_rgba[idx].copy()
    material = trimesh.visual.material.PBRMaterial(
      baseColorFactor=rgba,
      metallicFactor=0.5,
      roughnessFactor=0.5,
      emissiveFactor=[0.0, 0.0, 0.0],
    )
    if geom_type == mjtGeom.mjGEOM_PLANE:
      # Create a plane mesh
      out = trimesh.creation.box((20, 20, 0.01))
      out.visual = trimesh.visual.TextureVisuals(material=material)
      return out
    elif geom_type == mjtGeom.mjGEOM_SPHERE:
      radius = size[0]
      out = trimesh.creation.icosphere(radius=radius, subdivisions=2)
      out.visual = trimesh.visual.TextureVisuals(material=material)
      return out
    elif geom_type == mjtGeom.mjGEOM_BOX:
      dims = 2.0 * size
      out = trimesh.creation.box(extents=dims)
      out.visual = trimesh.visual.TextureVisuals(material=material)
      return out
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
      mesh.visual = trimesh.visual.TextureVisuals(material=material)
      return mesh

    elif geom_type == mjtGeom.mjGEOM_CAPSULE:
      r, half_len = size[0], size[1]
      out = trimesh.creation.capsule(radius=r, height=2.0 * half_len)
      out.visual = trimesh.visual.TextureVisuals(material=material)
      return out
    elif geom_type == mjtGeom.mjGEOM_CYLINDER:
      r, half_len = size[0], size[1]
      out = trimesh.creation.cylinder(radius=r, height=2.0 * half_len)
      out.visual = trimesh.visual.TextureVisuals(material=material)
      return out
    else:
      raise ValueError(f"Unsupported geom type {geom_type}")
