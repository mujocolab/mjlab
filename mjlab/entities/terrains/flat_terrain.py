import mujoco
from mjlab.entities.terrains import terrain_config


texture = terrain_config.TextureCfg(
  type="2d",
  builtin="checker",
  mark="edge",
  rgb1=(0.2, 0.3, 0.4),
  rgb2=(0.1, 0.2, 0.3),
  markrgb=(0.8, 0.8, 0.8),
  width=300,
  height=300,
)

material = terrain_config.MaterialCfg(
  texuniform=True,
  texrepeat=(4, 4),
  reflectance=0.2,
  texture="groundplane",
)

def add_floor(spec: mujoco.MjSpec) -> None:
  spec.worldbody.add_geom(
    name="floor",
    size=(0, 0, 0.01),
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    material="groundplane",
  )


FLAT_TERRAIN_CFG = terrain_config.TerrainCfg(
  textures={
    "groundplane": texture,
  },
  materials={
    "groundplane": material,
  },
  construct_fn=add_floor,
)