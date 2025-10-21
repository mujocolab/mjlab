"""Visualize NaN dump states in Viser.

Loads a NaN dump (npz + mjb) and provides a slider to scrub through captured
states frame by frame.

Example:
  uv run viz-nan /tmp/mjlab/nan_dumps/nan_dump_20251014_095857.npz
"""

from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser
import viser.transforms as vtf
from mujoco import mjtGeom

from mjlab.viewer.viser_conversions import get_body_name, is_fixed_body, merge_geoms


class NanDumpViewer:
  """Viewer for NaN dump files."""

  def __init__(self, dump_path: str | Path):
    self.dump_path = Path(dump_path)
    self.output_dir = self.dump_path.parent

    print(f"Loading NaN dump from: {self.dump_path}")
    self.dump = np.load(self.dump_path, allow_pickle=True)
    self.metadata = self.dump["_metadata"].item()

    model_file = self.output_dir / self.metadata["model_file"]
    print(f"Loading model from: {model_file}")
    self.model = mujoco.MjModel.from_binary_path(str(model_file))
    self.data = mujoco.MjData(self.model)

    self.state_keys = sorted(
      [k for k in self.dump.keys() if k.startswith("states_step_")],
      key=lambda x: int(x.split("_")[-1]),
    )
    self.num_steps = len(self.state_keys)
    self.num_envs_captured = self.metadata["num_envs_captured"]

    print("\nDump info:")
    print(f"  Total environments: {self.metadata['num_envs_total']}")
    print(f"  Captured environments: {self.num_envs_captured}")
    print(f"  NaN detected in envs: {self.metadata['nan_env_ids']}")
    print(f"  Buffer size: {self.num_steps} steps")
    print(f"  State size: {self.metadata['state_size']}")

    self.server = viser.ViserServer(label="NaN Dump Viewer")
    self.mesh_handles: dict[int, viser.GlbHandle] = {}

    self.current_step = 0
    self.current_env = 0

  def setup(self) -> None:
    """Setup the viewer GUI and scene."""
    with self.server.gui.add_folder("Info"):
      self.info_html = self.server.gui.add_html(self._get_info_html())

    with self.server.gui.add_folder("Controls"):
      self.step_slider = self.server.gui.add_slider(
        "Step",
        min=0,
        max=self.num_steps - 1,
        step=1,
        initial_value=0,
        hint=f"Scrub through {self.num_steps} captured states",
      )

      @self.step_slider.on_update
      def _(_) -> None:
        self.current_step = int(self.step_slider.value)
        self._update_state()

      if self.num_envs_captured > 1:
        self.env_slider = self.server.gui.add_slider(
          "Environment",
          min=0,
          max=self.num_envs_captured - 1,
          step=1,
          initial_value=0,
          hint=f"Select environment (0-{self.num_envs_captured - 1})",
        )

        @self.env_slider.on_update
        def _(_) -> None:
          self.current_env = int(self.env_slider.value)
          self._update_state()

    self._setup_scene()
    self._update_state()

  def _get_info_html(self) -> str:
    """Generate info HTML."""
    nan_env_ids = self.metadata["nan_env_ids"]
    nan_env_str = ", ".join(str(e) for e in nan_env_ids[:10])
    if len(nan_env_ids) > 10:
      nan_env_str += "..."

    step_name = self.state_keys[self.current_step]
    step_num = int(step_name.split("_")[-1])

    is_nan_env = self.current_env in nan_env_ids
    nan_indicator = "⚠️ NaN Detected" if is_nan_env else "✓ Clean"

    return f"""
      <div style="font-size: 0.85em; line-height: 1.25;">
        <strong>Step:</strong> {step_num}<br/>
        <strong>Environment:</strong> {self.current_env}<br/>
        <strong>Status:</strong> {nan_indicator}<br/>
        <strong>NaN envs:</strong> {nan_env_str}
      </div>
    """

  def _setup_scene(self) -> None:
    """Setup the 3D scene with the model."""
    self.server.scene.configure_environment_map(environment_intensity=0.8)
    self._add_fixed_geometry()
    self._create_body_meshes()

  def _add_fixed_geometry(self) -> None:
    """Add fixed world geometry to the scene."""
    for body_id in range(self.model.nbody):
      if not is_fixed_body(self.model, body_id):
        continue

      body_name = get_body_name(self.model, body_id)

      geom_ids = []
      for geom_id in range(self.model.ngeom):
        if self.model.geom_bodyid[geom_id] == body_id:
          is_collision = self.model.geom_contype[geom_id] != 0
          if is_collision and body_name != "terrain":
            continue
          geom_ids.append(geom_id)

      if not geom_ids:
        continue

      for geom_id in geom_ids:
        if self.model.geom_type[geom_id] == mjtGeom.mjGEOM_PLANE:
          self.server.scene.add_grid(
            f"/fixed/{body_name}/plane_{geom_id}",
            width=2000.0,
            height=2000.0,
            infinite_grid=True,
            fade_distance=50.0,
            position=self.model.geom_pos[geom_id],
            wxyz=self.model.geom_quat[geom_id],
          )
          geom_ids.remove(geom_id)
          break

      if geom_ids:
        mesh = merge_geoms(self.model, geom_ids)
        self.server.scene.add_mesh_trimesh(
          f"/fixed/{body_name}",
          mesh,
          position=self.model.body(body_id).pos,
          wxyz=self.model.body(body_id).quat,
        )

  def _create_body_meshes(self) -> None:
    """Create mesh handles for movable bodies."""
    for body_id in range(self.model.nbody):
      if is_fixed_body(self.model, body_id):
        continue

      body_name = get_body_name(self.model, body_id)

      geom_ids = []
      for geom_id in range(self.model.ngeom):
        if self.model.geom_bodyid[geom_id] == body_id:
          if self.model.geom_contype[geom_id] != 0:
            continue
          geom_ids.append(geom_id)

      if not geom_ids:
        continue

      mesh = merge_geoms(self.model, geom_ids)
      handle = self.server.scene.add_mesh_trimesh(
        f"/bodies/{body_name}",
        mesh,
        position=(0, 0, 0),
        wxyz=(1, 0, 0, 0),
      )
      self.mesh_handles[body_id] = handle

  def _update_state(self) -> None:
    """Update the visualization to show the current state."""
    state_key = self.state_keys[self.current_step]
    states = self.dump[state_key]
    state = states[self.current_env]

    mujoco.mj_setState(self.model, self.data, state, mujoco.mjtState.mjSTATE_PHYSICS)
    mujoco.mj_forward(self.model, self.data)

    for body_id, handle in self.mesh_handles.items():
      pos = self.data.xpos[body_id].copy()
      xmat = self.data.xmat[body_id].reshape(3, 3)
      quat = vtf.SO3.from_matrix(xmat).wxyz

      handle.position = pos
      handle.wxyz = quat

    self.info_html.content = self._get_info_html()

  def run(self) -> None:
    """Run the viewer (blocks until server is stopped)."""
    print("\nUse the sliders to scrub through states.")
    print("Press Ctrl+C to exit.")

    try:
      while True:
        import time

        time.sleep(0.1)
    except KeyboardInterrupt:
      print("\nShutting down...")
      self.server.stop()


def run_viewer(dump_path: tyro.conf.Positional[str]):
  """View NaN dump states in Viser.

  Args:
    dump_path: Path to nan_dump_TIMESTAMP.npz file.
  """
  viewer = NanDumpViewer(dump_path)
  viewer.setup()
  viewer.run()


def main():
  """CLI entry point for viz-nan command."""
  tyro.cli(run_viewer, description=__doc__)


if __name__ == "__main__":
  main()
