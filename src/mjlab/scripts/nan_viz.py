"""Visualize NaN dump states in Viser.

Loads a NaN dump (npz + mjb) and provides a slider to scrub through captured
states frame by frame.

Example:
  uv run viz-nan /tmp/mjlab/nan_dumps/nan_dump_20251014_095857.npz
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import tyro
import viser

import mjlab
from mjlab.utils.nan_replay import NanDump
from mjlab.viewer.viser import ViserMujocoScene


class NanDumpViewer:
  """Viewer for NaN dump files."""

  def __init__(self, dump_path: str | Path):
    self.dump_path = Path(dump_path)
    self.output_dir = self.dump_path.parent

    print(f"Loading NaN dump from: {self.dump_path}")
    self.archive = NanDump(self.dump_path)
    self.metadata = self.archive.metadata
    self.state_keys = self.archive.state_keys
    self.state_spec = self.archive.state_spec
    self.model, self.data = self.archive.restore(0, 0)
    self._state_lock = Lock()
    self.num_steps = len(self.state_keys)
    self.num_envs_dumped = self.metadata["num_envs_dumped"]
    self.dumped_env_ids = self.metadata["dumped_env_ids"]

    print("\nDump info:")
    print(f"  Total environments: {self.metadata['num_envs_total']}")
    print(f"  Dumped environments: {self.num_envs_dumped}")
    print(f"  Dumped env IDs: {self.dumped_env_ids}")
    print(f"  NaN detected in envs: {self.metadata['nan_env_ids']}")
    print(f"  Buffer size: {self.num_steps} steps")
    print(f"  State size: {self.metadata['state_size']}")

    self.server = viser.ViserServer(label="NaN Dump Viewer")
    self.current_step = 0
    self.current_env = 0
    self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)

  def setup(self) -> None:
    """Setup the viewer GUI and scene."""
    self.info_html = self.server.gui.add_html(self._get_info_html())

    with self.server.gui.add_folder("Playback"):
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

      if self.num_envs_dumped > 1:
        self.env_slider = self.server.gui.add_slider(
          "Environment",
          min=0,
          max=self.num_envs_dumped - 1,
          step=1,
          initial_value=0,
          hint=f"Select environment (0-{self.num_envs_dumped - 1})",
        )

        @self.env_slider.on_update
        def _(_) -> None:
          self.current_env = int(self.env_slider.value)
          self._update_state()

    # Add standard visualization options (hide debug viz control since no env).
    with self.server.gui.add_folder("Scene") as self.scene_folder:
      self.scene.create_scene_gui(show_debug_viz_control=False)

    # Initial state update.
    self._update_state()

  def _get_info_html(self) -> str:
    """Generate info HTML."""
    nan_env_ids = self.metadata["nan_env_ids"]
    nan_env_str = ", ".join(str(e) for e in nan_env_ids[:10])
    if len(nan_env_ids) > 10:
      nan_env_str += "..."

    step_name = self.state_keys[self.current_step]
    step_num = int(step_name.split("_")[-1])

    actual_env_id = self.dumped_env_ids[self.current_env]
    is_nan_env = actual_env_id in nan_env_ids
    nan_indicator = "⚠️ NaN Detected" if is_nan_env else "✓ Clean"

    return f"""
      <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
        <strong>Step:</strong> {step_num}<br/>
        <strong>Environment:</strong> {actual_env_id}<br/>
        <strong>Status:</strong> {nan_indicator}<br/>
        <strong>NaN envs:</strong> {nan_env_str}
      </div>
    """

  def _update_state(self) -> None:
    """Update the visualization to show the current state."""
    with self._state_lock, self.server.atomic():
      model, data = self.archive.restore(
        self.current_step, self.current_env, forward=True
      )
      if model is not self.model:
        # Geometry/material variants require new mesh handles and scene controls.
        self.scene_folder.remove()
        self.server.scene.reset()
        self.scene = ViserMujocoScene(self.server, model, num_envs=1)
        with self.server.gui.add_folder("Scene") as self.scene_folder:
          self.scene.create_scene_gui(show_debug_viz_control=False)
      self.model, self.data = model, data
      self.scene.update_from_mjdata(self.data)
      self.info_html.content = self._get_info_html()

  def run(self) -> None:
    """Run the viewer (blocks until server is stopped)."""
    print("\nUse the sliders to scrub through states.")
    print("Press Ctrl+C to exit.")

    try:
      while True:
        import time

        # Check if visualization settings changed and need a refresh.
        with self._state_lock:
          if self.scene.needs_update:
            self.scene.refresh_visualization()

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

  tyro.cli(run_viewer, description=__doc__, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
