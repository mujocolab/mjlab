"""Mjlab viewer based on Viser.

Adapted from an MJX visualizer by Chung Min Kim: https://github.com/chungmin99/
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import numpy as np
import trimesh
import trimesh.visual
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj  # type: ignore
from typing_extensions import assert_never, override

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
    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._batch_size = self.env.num_envs

    self._counter = 0
    self._env_idx = 0
    self._show_only_selected_env = (
      False  # Track whether to show only selected environment
    )

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

      # Environment selection moved to Reward Plots tab

      # Display settings
      with self._server.gui.add_folder("Display Settings"):
        cb_collision = self._server.gui.add_checkbox(
          "Collision geom", initial_value=False
        )
        cb_visual = self._server.gui.add_checkbox("Visual geom", initial_value=True)

        @cb_collision.on_update
        def _(_) -> None:
          if cb_collision.value:
            self._ensure_collision_handles_exist()

          if self._mesh_collision_handles is not None:
            for handle in self._mesh_collision_handles.values():
              handle.visible = cb_collision.value

        @cb_visual.on_update
        def _(_) -> None:
          if cb_visual.value:
            self._ensure_visual_handles_exist()

          if self._mesh_visual_handles is not None:
            for handle in self._mesh_visual_handles.values():
              handle.visible = cb_visual.value

    # Reward plots tab
    if hasattr(self.env.unwrapped, "reward_manager"):
      with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
        # Environment selection if multiple environments
        if self.env.num_envs > 1:
          with self._server.gui.add_folder("Environment Selection"):
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
              "Select Environment",
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

        # Get reward term names and create reward plotter
        term_names = [
          name
          for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self._env_idx
          )
        ]
        self._reward_plotter = ViserRewardPlotter(self._server, term_names)

    # Get initial geometry positions to find natural center for floor grid
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    # wp_data = sim.wp_data
    # geom_xpos = wp_data.geom_xpos.numpy()  # Shape: (batch_size, ngeom, 3)

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
            self._server.scene.add_mesh_trimesh(
              f"/fixed_bodies/{body_name}/{visual_or_collision}",
              self._merge_geoms(mj_model, nonplane_geom_ids),
              cast_shadow=False,
              receive_shadow=0.2,
            )
      # Dynamic bodies - skip creation, will be handled lazily
      else:
        pass

    # Create visual handles by default on startup
    self._ensure_visual_handles_exist()

    # Set up per-client camera controls
    @self._server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
      """Set up camera controls for each connected client."""
      with client.gui.add_folder("Camera"):
        fov_slider = client.gui.add_slider(
          "Field of View (degrees)",
          min=20,
          max=120,
          step=1,
          initial_value=90,
        )

        @fov_slider.on_update
        def _(_) -> None:
          client.camera.fov = np.radians(fov_slider.value)

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

      # Determine if this geom should be included based on mesh_type
      should_include: bool
      if mesh_type == "collision":
        should_include = is_collision
      elif mesh_type == "visual":
        should_include = not is_collision
      else:
        assert_never(mesh_type)

      # Add geom to body's list if it matches the type we're looking for
      if should_include:
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

    # We'll make a copy of the relevant state, then do the update itself asynchronously.
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    mj_model = sim.mj_model
    wp_data = sim.wp_data

    geom_xpos = wp_data.geom_xpos.numpy()
    assert geom_xpos.shape == (self._batch_size, mj_model.ngeom, 3)
    geom_xmat = wp_data.geom_xmat.numpy()
    assert geom_xmat.shape == (self._batch_size, mj_model.ngeom, 3, 3)

    # Get body positions and orientations from warp data
    body_xpos = wp_data.xpos.numpy()  # Shape: (batch_size, nbody, 3)
    body_xmat = wp_data.xmat.numpy()  # Shape: (batch_size, nbody, 3, 3)

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
              single_quat = body_xquat[self._env_idx, body_id, :]
              # Replicate single environment data for all batch slots
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
