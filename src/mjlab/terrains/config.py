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
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 1.0),
      platform_width=2.0,
      border_width=0.25,
    ),
    "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 1.0),
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
  },
  add_lights=True,
)

ALL_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=16,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=1.0),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.0, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.0, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=1.0,
      slope_range=(0.0, 0.7),
      platform_width=2.0,
      border_width=0.25,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
    "wave_terrain": terrain_gen.HfWaveTerrainCfg(
      proportion=1.0,
      amplitude_range=(0.0, 0.2),
      num_waves=4,
      border_width=0.25,
    ),
    "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
      proportion=1.0,
      obstacle_width_range=(0.3, 1.0),
      obstacle_height_range=(0.05, 0.3),
      num_obstacles=40,
      border_width=0.25,
    ),
    "perlin_noise": terrain_gen.HfPerlinNoiseTerrainCfg(
      proportion=1.0,
      height_range=(0.0, 1.0),
      octaves=4,
      persistence=0.3,
      lacunarity=2.0,
      scale=10.0,
      horizontal_scale=0.1,
      border_width=0.50,
    ),
    "box_random_grid": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=1.0,
      grid_width=0.4,
      grid_height_range=(0.0, 0.3),
      platform_width=1.0,
    ),
    "random_spread_boxes": terrain_gen.BoxRandomSpreadTerrainCfg(
      proportion=1.0,
      num_boxes=80,
      box_width_range=(0.1, 1.0),
      box_length_range=(0.1, 2.0),
      box_height_range=(0.05, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
    "open_stairs": terrain_gen.BoxOpenStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.1, 0.2),
      step_width_range=(0.4, 0.8),
      platform_width=1.0,
      border_width=0.25,
    ),
    "random_stairs": terrain_gen.BoxRandomStairsTerrainCfg(
      proportion=1.0,
      step_width=0.8,
      step_height_range=(0.1, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
    "stepping_stones": terrain_gen.BoxSteppingStonesTerrainCfg(
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
    "narrow_beams": terrain_gen.BoxNarrowBeamsTerrainCfg(
      proportion=1.0,
      num_beams=12,
      beam_width_range=(0.2, 0.8),
      beam_height=0.2,
      spacing=0.8,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
    "nested_rings": terrain_gen.BoxNestedRingsTerrainCfg(
      proportion=1.0,
      num_rings=8,
      ring_width_range=(0.3, 0.6),
      gap_range=(0.1, 0.4),
      height_range=(0.1, 0.4),
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
    "tilted_grid": terrain_gen.BoxTiltedGridTerrainCfg(
      proportion=1.0,
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
