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
    "flat": terrain_gen.BoxFlatTerrainCfg(
      proportion=1 / 3,
    ),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=1 / 3,
      step_height_range=(0.05, 0.23),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
      holes=False,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=1 / 3,
      step_height_range=(0.05, 0.23),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
      holes=False,
    ),
    # "boxes": terrain_gen.BoxRandomGridTerrainCfg(
    #   proportion=1 / 3,
    #   grid_width=0.45,
    #   grid_height_range=(0.05, 0.2),
    #   platform_width=2.0,
    # ),
  },
)


if __name__ == "__main__":
  import mujoco.viewer

  terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
  )
  terrain = TerrainImporter(terrain_cfg, device="cuda:0")
  mujoco.viewer.launch(terrain.spec.compile())
