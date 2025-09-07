import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  horizontal_scale=0.1,
  vertical_scale=0.005,
  slope_threshold=0.75,
  use_cache=False,
  sub_terrains={
    "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.05, 0.23),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
      holes=False,
    ),
    "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.05, 0.23),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
      holes=False,
    ),
    "boxes": terrain_gen.MeshRandomGridTerrainCfg(
      proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
    ),
  },
)

if __name__ == "__main__":
  from mjlab.terrains.terrain_importer import TerrainImporter
  from mjlab.terrains.terrain_importer_cfg import TerrainImporterCfg

  terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
    collision_group=-1,
  )

  terrain = TerrainImporter(terrain_cfg)

  import mujoco.viewer

  mujoco.viewer.launch(terrain._spec.compile())
