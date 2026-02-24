"""Render images of each sub-terrain type for documentation.

Generates one PNG per terrain type at a fixed difficulty level.
Images are saved to docs/source/_static/terrains/.

Run with:
  uv run python scripts/tools/render_terrain_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import mjlab.terrains as terrain_gen
from mjlab.terrains import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils import spec_config as spec_cfg

OUTPUT_DIR = Path("docs/source/_static/terrains")
WIDTH = 1080
HEIGHT = 1080
DIFFICULTY = 0.85
PATCH_SIZE = (8.0, 8.0)


# Each entry: (filename, SubTerrainCfg instance).
TERRAIN_TYPES: list[tuple[str, terrain_gen.SubTerrainCfg]] = [
  # Primitive terrains.
  (
    "box_flat",
    terrain_gen.BoxFlatTerrainCfg(proportion=1.0),
  ),
  (
    "box_pyramid_stairs",
    terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.0, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
  ),
  (
    "box_inverted_pyramid_stairs",
    terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.0, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
  ),
  (
    "box_random_stairs",
    terrain_gen.BoxRandomStairsTerrainCfg(
      proportion=1.0,
      step_width=0.8,
      step_height_range=(0.1, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
  ),
  (
    "box_open_stairs",
    terrain_gen.BoxOpenStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.1, 0.2),
      step_width_range=(0.4, 0.8),
      platform_width=1.0,
      border_width=0.25,
    ),
  ),
  (
    "box_random_grid",
    terrain_gen.BoxRandomGridTerrainCfg(
      proportion=1.0,
      grid_width=0.4,
      grid_height_range=(0.0, 0.3),
      platform_width=1.0,
    ),
  ),
  (
    "box_random_spread",
    terrain_gen.BoxRandomSpreadTerrainCfg(
      proportion=1.0,
      num_boxes=80,
      box_width_range=(0.1, 1.0),
      box_length_range=(0.1, 2.0),
      box_height_range=(0.05, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
  ),
  (
    "box_stepping_stones",
    terrain_gen.BoxSteppingStonesTerrainCfg(
      proportion=1.0,
      stone_size_range=(0.4, 0.8),
      stone_distance_range=(0.2, 0.5),
      stone_height=0.2,
      stone_height_variation=0.1,
      stone_size_variation=0.2,
      displacement_range=0.1,
      floor_depth=2.0,
      platform_width=1.0,
      border_width=0.25,
    ),
  ),
  (
    "box_narrow_beams",
    terrain_gen.BoxNarrowBeamsTerrainCfg(
      proportion=1.0,
      num_beams=12,
      beam_width_range=(0.2, 0.8),
      beam_height=0.2,
      spacing=0.8,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
  ),
  (
    "box_tilted_grid",
    terrain_gen.BoxTiltedGridTerrainCfg(
      proportion=1.0,
      grid_width=1.0,
      tilt_range_deg=20.0,
      height_range=0.3,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
  ),
  (
    "box_nested_rings",
    terrain_gen.BoxNestedRingsTerrainCfg(
      proportion=1.0,
      num_rings=8,
      ring_width_range=(0.3, 0.6),
      gap_range=(0.1, 0.4),
      height_range=(0.1, 0.4),
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
  ),
  # Heightfield terrains.
  (
    "hf_pyramid_slope",
    terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=1.0,
      slope_range=(0.0, 0.7),
      platform_width=2.0,
      border_width=0.25,
    ),
  ),
  (
    "hf_random_uniform",
    terrain_gen.HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
  ),
  (
    "hf_wave",
    terrain_gen.HfWaveTerrainCfg(
      proportion=1.0,
      amplitude_range=(0.1, 0.5),
      num_waves=6,
      border_width=0.25,
    ),
  ),
  (
    "hf_discrete_obstacles",
    terrain_gen.HfDiscreteObstaclesTerrainCfg(
      proportion=1.0,
      obstacle_width_range=(0.3, 1.0),
      obstacle_height_range=(0.05, 0.3),
      num_obstacles=40,
      border_width=0.25,
    ),
  ),
  (
    "hf_perlin_noise",
    terrain_gen.HfPerlinNoiseTerrainCfg(
      proportion=1.0,
      height_range=(0.0, 1.0),
      octaves=4,
      persistence=0.3,
      lacunarity=2.0,
      scale=10.0,
      horizontal_scale=0.1,
      border_width=0.50,
    ),
  ),
]


CAMERA_DISTANCE = 8.0
CAMERA_ELEVATION_DEG = 50.0  # Degrees from horizontal (90 = top-down).
CAMERA_AZIMUTH_DEG = 135.0  # Rotation around the vertical axis.
FOV_PADDING = 1.1


def render_terrain(
  name: str,
  sub_terrain_cfg: terrain_gen.SubTerrainCfg,
) -> np.ndarray:
  """Generate and render a single terrain type, return RGB array."""
  terrain_cfg = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=42,
      size=PATCH_SIZE,
      num_rows=1,
      num_cols=1,
      border_width=0.0,
      curriculum=False,
      difficulty_range=(DIFFICULTY, DIFFICULTY),
      color_scheme="height",
      sub_terrains={name: sub_terrain_cfg},
    ),
    lights=(
      spec_cfg.LightCfg(
        name="sun",
        type="directional",
        dir=(-0.15, -0.15, -0.97),
        castshadow=True,
      ),
    ),
  )
  terrain = TerrainEntity(terrain_cfg, device="cpu")

  # Place camera at an isometric-ish angle to reveal 3D structure.
  elev = np.deg2rad(CAMERA_ELEVATION_DEG)
  azim = np.deg2rad(CAMERA_AZIMUTH_DEG)
  cam_pos = CAMERA_DISTANCE * np.array(
    [
      np.cos(elev) * np.cos(azim),
      np.cos(elev) * np.sin(azim),
      np.sin(elev),
    ]
  )

  # Look at a point slightly below the origin so the terrain centers
  # in the frame instead of sitting in the bottom half.
  lookat = np.array([0.0, 0.0, -2.0])
  forward = lookat - cam_pos
  forward /= np.linalg.norm(forward)
  # MuJoCo camera convention: -z is forward, y is up.
  up = np.array([0.0, 0.0, 1.0])
  right = np.cross(forward, up)
  right /= np.linalg.norm(right)
  cam_up = np.cross(right, forward)

  # Build rotation matrix (columns: right, cam_up, -forward).
  rot = np.column_stack([right, cam_up, -forward])
  # Convert to quaternion (wxyz).
  from scipy.spatial.transform import Rotation

  quat_xyzw = Rotation.from_matrix(rot).as_quat()
  quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

  # FOV sized to fit the patch diagonal from this viewing angle.
  apparent_size = max(PATCH_SIZE) * FOV_PADDING
  fovy_rad = 2 * np.arctan2(apparent_size / 2, CAMERA_DISTANCE)

  terrain.spec.worldbody.add_camera(
    name="gallery",
    pos=cam_pos.tolist(),
    quat=quat_wxyz,
    fovy=np.rad2deg(fovy_rad),
  )

  model = terrain.spec.compile()
  model.vis.global_.offheight = HEIGHT
  model.vis.global_.offwidth = WIDTH
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
    renderer.update_scene(data, camera="gallery")
    return renderer.render()


def main() -> None:
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  for filename, sub_cfg in TERRAIN_TYPES:
    print(f"Rendering {filename}...")
    img = render_terrain(filename, sub_cfg)
    Image.fromarray(img).save(OUTPUT_DIR / f"{filename}.png")
    print(f"  Saved {OUTPUT_DIR / filename}.png")

  print(f"\nDone. {len(TERRAIN_TYPES)} images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
  main()
