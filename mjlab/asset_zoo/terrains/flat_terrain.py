from mjlab.entities.terrains import terrain_config
from mjlab.utils.spec_editor import spec_editor_config

texture = spec_editor_config.TextureCfg(
  name="groundplane",
  type="2d",
  builtin="checker",
  mark="edge",
  rgb1=(0.2, 0.3, 0.4),
  rgb2=(0.1, 0.2, 0.3),
  markrgb=(0.8, 0.8, 0.8),
  width=300,
  height=300,
)

material = spec_editor_config.MaterialCfg(
  name="groundplane",
  texuniform=True,
  texrepeat=(4, 4),
  reflectance=0.2,
  texture="groundplane",
)

geom = spec_editor_config.GeomCfg(
  name="floor",
  type="plane",
  size=(0, 0, 0.01),
  material="groundplane",
)

FLAT_TERRAIN_CFG = terrain_config.TerrainCfg(
  textures=[
    texture,
  ],
  materials=[
    material,
  ],
  geoms=[
    geom,
  ],
)

if __name__ == "__main__":
  from mjlab.entities.terrains.terrain import Terrain
  import mujoco.viewer

  terr = Terrain(FLAT_TERRAIN_CFG)
  mujoco.viewer.launch(terr.compile())
