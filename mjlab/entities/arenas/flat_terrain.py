import mujoco

from mjlab.entities.arenas import arena


class FlatTerrainArena(arena.Arena):
  """An arena with a flat terrain."""

  def post_init(self):
    self.spec.add_texture(
      name="groundplane",
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
      mark=mujoco.mjtMark.mjMARK_EDGE,
      rgb1=(0.2, 0.3, 0.4),
      rgb2=(0.1, 0.2, 0.3),
      markrgb=(0.8, 0.8, 0.8),
      width=300,
      height=300,
    )
    self.spec.add_material(
      name="groundplane",
      texuniform=True,
      texrepeat=(4, 4),
      reflectance=0.2,
    ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

    self._floor_geom = self.spec.worldbody.add_geom(
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
