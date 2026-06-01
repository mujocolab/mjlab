"""Interactive single-patch terrain explorer (Viser + MuJoCo MjSpec).

Run with:
  uv run python scripts/tools/terrain_explorer.py
  uv run python scripts/tools/terrain_explorer.py --port 8081

Then open the printed URL (default http://localhost:8080).
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
import viser
from mjviser.conversions import merge_geoms

from mjlab.terrains.config import ALL_TERRAIN_PRESETS
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg

PATCH_SIZE = (8.0, 8.0)


# Per-preset overrides applied when building in the explorer (e.g. to surface
# difficulty-driven behavior that is off by default).
_PRESET_OVERRIDES: dict[str, dict] = {
  "random_rough": {"scale_with_difficulty": True},
}


def _build_terrain_mesh(preset_name: str, difficulty: float, seed: int):
  """Generate a single terrain patch and return a merged trimesh (or raise)."""
  preset_fn = ALL_TERRAIN_PRESETS[preset_name]
  overrides = _PRESET_OVERRIDES.get(preset_name, {})
  generator_cfg = TerrainGeneratorCfg(
    seed=seed,
    size=PATCH_SIZE,
    num_rows=1,
    num_cols=1,
    border_width=0.0,
    curriculum=False,
    # A degenerate range pins the single patch to exactly this difficulty.
    difficulty_range=(difficulty, difficulty),
    color_scheme="height",
    sub_terrains={preset_name: preset_fn(proportion=1.0, **overrides)},
  )
  generator = TerrainGenerator(generator_cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()

  terrain_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
  geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == terrain_body_id]
  return merge_geoms(model, geom_ids)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--port", type=int, default=8080, help="Port for the viser server."
  )
  args = parser.parse_args()

  server = viser.ViserServer(port=args.port)
  preset_names = sorted(ALL_TERRAIN_PRESETS)

  terrain_dropdown = server.gui.add_dropdown(
    "Terrain", options=preset_names, initial_value=preset_names[0]
  )
  difficulty_slider = server.gui.add_slider(
    "Difficulty", min=0.0, max=1.0, step=0.01, initial_value=0.0
  )
  seed_input = server.gui.add_number("Seed", initial_value=42, step=1)
  status = server.gui.add_markdown("**Status:** ready")

  handle: viser.SceneNodeHandle | None = None

  def update() -> None:
    nonlocal handle
    name = terrain_dropdown.value
    difficulty = float(difficulty_slider.value)
    seed = int(seed_input.value)
    status.content = f"**Status:** building `{name}` at difficulty {difficulty:.2f}..."
    try:
      mesh = _build_terrain_mesh(name, difficulty, seed)
    except Exception as e:  # noqa: BLE001 - surface any generation failure in the UI.
      status.content = f"**Error:** {type(e).__name__}: {e}"
      print(f"Failed to build {name} at difficulty {difficulty}: {e}")
      return
    if handle is not None:
      handle.remove()
    handle = server.scene.add_mesh_trimesh("/terrain", mesh)
    status.content = (
      f"**Loaded** `{name}` at difficulty {difficulty:.2f} ({len(mesh.faces):,} faces)"
    )

  terrain_dropdown.on_update(lambda _: update())
  difficulty_slider.on_update(lambda _: update())
  seed_input.on_update(lambda _: update())

  # Top-down-ish initial camera.
  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = np.array([10.0, 10.0, 8.0])
    client.camera.look_at = np.array([0.0, 0.0, 0.0])

  update()
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()
