import mujoco

from mjlab._src.arenas import arena


class FlatTerrainArena(arena.Arena):
  """An arena with a flat terrain."""

  def __init__(self):
    super().__init__()

    self._spec.add_texture(
      name="groundplane",
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
      mark=mujoco.mjtMark.mjMARK_EDGE,
      rgb1=(1, 1, 1),
      rgb2=(1, 1, 1),
      markrgb=(0, 0, 0),
      width=300,
      height=300,
    )
    self._spec.add_material(
      name="groundplane",
      texuniform=True,
      texrepeat=(5, 5),
      reflectance=0,
    ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    self._floor_geom = self._spec.worldbody.add_geom(
      name="floor",
      size=(0, 0, 0.01),
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      material="groundplane",
    )

  @property
  def floor_geom(self) -> mujoco.MjsGeom:
    """The floor geometry."""
    return self._floor_geom


if __name__ == "__main__":
  import mujoco.viewer

  scene = FlatTerrainArena()
  scene.add_skybox()

  mujoco.viewer.launch(scene.compile())
