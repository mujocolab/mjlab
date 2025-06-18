from typing import Dict
import mujoco

from mjlab import MJLAB_SRC_PATH, update_assets
from mjlab.entities.arenas import arena

_ASSETS_DIR = MJLAB_SRC_PATH / "entities" / "arenas" / "xmls" / "assets"


def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  update_assets(assets, _ASSETS_DIR)
  return assets


class PlaygroundTerrainArena(arena.Arena):
  """Hfield terrain used in MuJoCo Playground."""

  def __init__(self):
    super().__init__(assets=get_assets())

  def post_init(self):
    self.spec.add_hfield(
      name="hfield",
      file="hfield.png",
      size=(10, 10, 0.05, 0.1),
    )
    self.spec.add_texture(
      name="groundplane",
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      file="rocky_texture.png",
    )
    self.spec.add_material(
      name="groundplane",
      texuniform=True,
      texrepeat=(5, 5),
    ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

    self._floor_geom = self.spec.worldbody.add_geom(
      name="floor",
      size=(0, 0, 0.01),
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname="hfield",
      material="groundplane",
    )

  @property
  def floor_geom(self) -> mujoco.MjsGeom:
    """The floor geometry."""
    return self._floor_geom


if __name__ == "__main__":
  import mujoco.viewer

  scene = PlaygroundTerrainArena()
  scene.add_skybox()

  mujoco.viewer.launch(scene.compile())
