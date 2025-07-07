import mujoco

from mjlab.core import entity
from mjlab.entities.terrains.terrain_config import TerrainCfg
from mjlab.entities.terrains import editors


class Terrain(entity.Entity):
  def __init__(self, terrain_cfg: TerrainCfg):
    self._cfg = terrain_cfg

    assets = terrain_cfg.asset_fn()
    if terrain_cfg.xml_path is not None:
      self._spec = mujoco.MjSpec.from_file(str(terrain_cfg.xml_path), assets=assets)
    else:
      self._spec = mujoco.MjSpec()
      self._spec.assets = assets

    self._configure_textures()
    self._configure_materials()
    self._configure_spec()

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  # Private methods.

  def _configure_textures(self) -> None:
    for tex_name, tex in self._cfg.textures.items():
      editors.TextureEditor(tex_name, tex).edit_spec(self._spec)

  def _configure_materials(self) -> None:
    for mat_name, mat in self._cfg.materials.items():
      editors.MaterialEditor(mat_name, mat).edit_spec(self._spec)

  def _configure_spec(self):
    if self._cfg.construct_fn is not None:
      self._cfg.construct_fn(self._spec)
