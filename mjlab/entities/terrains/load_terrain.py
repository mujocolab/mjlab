import mujoco.viewer

from mjlab.entities.terrains.terrain import Terrain
from mjlab.entities.terrains.flat_terrain import FLAT_TERRAIN_CFG

if __name__ == "__main__":
  terrain = Terrain(FLAT_TERRAIN_CFG)
  mujoco.viewer.launch(terrain.compile())
