"""Mjlab viewer based on Viser.

Adapted from an MJX visualizer by Chung Min Kim: https://github.com/chungmin99/
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import trimesh
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
    env_spacing: float = 1.0,
  ) -> None:
    super().__init__(env, policy, frame_rate, render_all_envs, verbosity)
    self._reward_plotter: Optional[ViserRewardPlotter] = None
    self._env_spacing = env_spacing

  @override
  def setup(self) -> None:
    """Setup the viewer resources."""

    self._server = viser.ViserServer(label="mjlab")
    self._handles: dict[
      str, tuple[viser.BatchedMeshHandle | viser.BatchedGlbHandle, bool]
    ] = {}
    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._batch_size = self.env.num_envs

    # Create grid layout for multiple environments
    # Environments are arranged in a square grid centered at origin
    cols = int(np.ceil(np.sqrt(self._batch_size)))
    rows = int(np.ceil(self._batch_size / cols))

    # Use meshgrid to create grid positions centered at origin
    x = np.arange(cols) * self._env_spacing - (cols - 1) * self._env_spacing / 2.0
    y = np.arange(rows) * self._env_spacing - (rows - 1) * self._env_spacing / 2.0
    xx, yy = np.meshgrid(x, y)

    # Flatten and stack to get offsets for all environments
    grid_positions = np.stack(
      [xx.flatten(), yy.flatten(), np.zeros(rows * cols)], axis=-1
    )
    # Only take the first batch_size positions (in case grid is larger than needed)
    self._env_offsets = grid_positions[: self._batch_size].astype(np.float32)
    self._counter = 0
    self._env_idx = 0
    self._show_only_selected_env = (
      False  # Track whether to show only selected environment
    )

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
      with self._server.gui.add_folder("Simulation Controls"):
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
        cb_floor = self._server.gui.add_checkbox("Floor geom", initial_value=False)

        # Slider to control spacing between environments (only show if multiple envs)
        if self.env.num_envs > 1:
          self._spacing_slider = self._server.gui.add_slider(
            "Environment Spacing",
            min=0.0,
            max=5.0,
            step=0.01,
            initial_value=self._env_spacing,
          )

          @self._spacing_slider.on_update
          def _(_) -> None:
            self._env_spacing = self._spacing_slider.value
            # Recalculate grid offsets with new spacing
            cols = int(np.ceil(np.sqrt(self._batch_size)))
            rows = int(np.ceil(self._batch_size / cols))

            # Use meshgrid to create grid positions centered at origin
            x = (
              np.arange(cols) * self._env_spacing - (cols - 1) * self._env_spacing / 2.0
            )
            y = (
              np.arange(rows) * self._env_spacing - (rows - 1) * self._env_spacing / 2.0
            )
            xx, yy = np.meshgrid(x, y)

            # Flatten and stack to get offsets for all environments
            grid_positions = np.stack(
              [xx.flatten(), yy.flatten(), np.zeros(rows * cols)], axis=-1
            )
            self._env_offsets = grid_positions[: self._batch_size].astype(np.float32)

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

        # Create reward plotter
        self._reward_plotter = ViserRewardPlotter(self._server)
        self._init_reward_plots()

    # Get initial geometry positions to find natural center for floor grid
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    wp_data = sim.wp_data
    geom_xpos = wp_data.geom_xpos.numpy()  # Shape: (batch_size, ngeom, 3)

    # Calculate average position across all environments and geometries
    avg_pos = geom_xpos.mean(axis=(0, 1))  # Average over envs and geoms
    floor_center_x, floor_center_y = avg_pos[0], avg_pos[1]

    # Add floor grid centered at the average geometry position
    self._server.scene.add_grid(
      "/floor", width=20.0, height=20.0, position=(floor_center_x, floor_center_y, 0.0)
    )
    for i in range(mj_model.ngeom):
      # Get geom properties
      name = mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, i)
      if not name:
        name = f"geom_{i}"

      pos = mj_model.geom_pos[i]
      quat = mj_model.geom_quat[i]  # (w, x, y, z)

      # Set color based on whether it's a collision geom
      is_collision = mj_model.geom_contype[i] != 0 or mj_model.geom_conaffinity[i] != 0

      # Get geometry type
      geom_type = mj_model.geom_type[i]

      # For mesh geometries, use the new conversion function
      if geom_type == mjtGeom.mjGEOM_MESH:
        # Convert MuJoCo mesh to trimesh with materials/textures
        mesh = mujoco_mesh_to_trimesh(mj_model, i, verbose=False)

        # Use add_batched_meshes_trimesh for better material/texture support
        handle = self._server.scene.add_batched_meshes_trimesh(
          f"/geoms/{name}",
          mesh,
          batched_wxyzs=quat[None].repeat(self._batch_size, axis=0),
          batched_positions=pos[None].repeat(self._batch_size, axis=0),
          lod="auto",
          visible="floor" not in name and not is_collision,
        )
      else:
        # For primitive geometries, use the existing approach
        mesh = self._create_mesh(mj_model, i)
        handle = self._server.scene.add_batched_meshes_simple(
          f"/geoms/{name}",
          vertices=mesh.vertices,
          faces=mesh.faces,
          batched_colors=(200, 100, 100) if is_collision else (30, 125, 230),
          batched_wxyzs=quat[None].repeat(self._batch_size, axis=0),
          batched_positions=pos[None].repeat(self._batch_size, axis=0),
          lod="auto",
          visible="floor" not in name and not is_collision,
        )

      self._handles[name] = (handle, is_collision)

  @override
  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""

    # Update counter
    self._counter += 1

    # Update status display less frequently (every 10 frames = ~6 updates per second)
    if self._counter % 10 == 0:
      self._update_status_display()

    # Skip every other frame for position updates to reduce load. 30FPS is fine!
    if self._counter % 2 == 0:
      return

    # Update reward plots
    if self._reward_plotter and not self._is_paused:
      terms = list(
        self.env.unwrapped.reward_manager.get_active_iterable_terms(self._env_idx)
      )
      self._reward_plotter.update(terms)

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

          if self._show_only_selected_env and self.env.num_envs > 1:
            # Show only the selected environment at the origin (0,0,0)
            single_pos = geom_xpos[self._env_idx, i, :]  # No offset - keep at origin
            single_quat = geom_xquat[self._env_idx, i, :]
            # Replicate single environment data for all batch slots
            handle.batched_positions = np.tile(
              single_pos[None, :], (self._batch_size, 1)
            )
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self._batch_size, 1))
          else:
            # Show all environments with offsets
            handle.batched_positions = geom_xpos[..., i, :] + self._env_offsets[:, :]
            handle.batched_wxyzs = geom_xquat[..., i, :]

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

  def _init_reward_plots(self) -> None:
    """Initialize reward plots."""
    if not self._reward_plotter:
      return

    # Get reward term names for current environment
    term_names = [
      name
      for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
        self._env_idx
      )
    ]

    self._reward_plotter.initialize(term_names)

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
