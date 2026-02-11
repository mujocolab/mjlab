import mujoco

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.2),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.0, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.0, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.7),
      platform_width=2.0,
      border_width=0.25,
    ),
    "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.7),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.1,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
    "wave_terrain": terrain_gen.HfWaveTerrainCfg(
      proportion=0.1,
      amplitude_range=(0.0, 0.2),
      num_waves=4,
      border_width=0.25,
    ),
    "box_random_grid": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=0.1,
      grid_width=0.4,
      grid_height_range=(0.0, 0.3),
      platform_width=1.0,
      holes=False,
      merge_similar_heights=False,
      height_merge_threshold=0.10,
      max_merge_distance=3,
    ),
    "box_random_grid_large": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=0.1,
      grid_width=0.8,
      grid_height_range=(0.0, 0.3),
      platform_width=1.0,
      holes=False,
      merge_similar_heights=False,
      height_merge_threshold=0.10,
      max_merge_distance=3,
    ),
    "perlin_terrain_smooth": terrain_gen.HfPerlinNoiseTerrainCfg(
      proportion=0.1,
      height_range=(0.0, 1.0),
      octaves=4,
      persistence=0.2,
      lacunarity=1.0,
      scale=5.0,
      horizontal_scale=0.1,
      base_thickness_ratio=1.0,
      border_width=0.50,
    ),
    "perlin_terrain_rough": terrain_gen.HfPerlinNoiseTerrainCfg(
      proportion=0.1,
      height_range=(0.0, 1.0),
      octaves=6,
      persistence=0.3,
      lacunarity=4.0,
      scale=10.0,
      horizontal_scale=0.1,
      base_thickness_ratio=1.0,
      border_width=0.50,
    ),
    "random_spread_boxes": terrain_gen.BoxRandomSpreadTerrainCfg(
      proportion=0.05,
      num_boxes=80,
      box_width_range=(0.1, 1.0),
      box_length_range=(0.1, 2.0),
      box_height_range=(0.05, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
    "open_stairs": terrain_gen.BoxOpenStairsTerrainCfg(
      proportion=0.05,
      step_height_range=(0.1, 0.2),
      step_width_range=(0.4, 0.8),
      platform_width=1.0,
      border_width=0.25,
      inverted=False,
    ),
    "inverted_open_stairs": terrain_gen.BoxOpenStairsTerrainCfg(
      proportion=0.05,
      step_height_range=(0.1, 0.2),
      step_width_range=(0.4, 0.8),
      platform_width=1.0,
      border_width=0.25,
      inverted=True,
    ),
    "random_stairs": terrain_gen.BoxRandomStairsTerrainCfg(
      proportion=0.05,
      step_width=0.8,
      step_height_range=(0.1, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
    "stepping_stones": terrain_gen.BoxSteppingStonesTerrainCfg(
      proportion=0.05,
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
    "narrow_beams": terrain_gen.BoxNarrowBeamsTerrainCfg(
      proportion=0.05,
      num_beams=12,
      beam_width_range=(0.2, 0.8),
      beam_height=0.2,
      spacing=0.8,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
    "nested_rings": terrain_gen.BoxNestedRingsTerrainCfg(
      proportion=0.05,
      num_rings=8,
      ring_width_range=(0.3, 0.6),
      gap_range=(0.1, 0.4),
      height_range=(0.1, 0.4),
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
    "tilted_grid": terrain_gen.BoxTiltedGridTerrainCfg(
      proportion=0.05,
      grid_width=1.0,
      tilt_range_deg=20.0,
      height_range=0.3,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
  },
  add_lights=True,
)


if __name__ == "__main__":
  import mujoco.viewer
  import torch

  device = "cuda" if torch.cuda.is_available() else "cpu"

  terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
  )
  terrain = TerrainImporter(terrain_cfg, device=device)
  mujoco.viewer.launch(terrain.spec.compile())
