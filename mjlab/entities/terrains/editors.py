from dataclasses import dataclass
import mujoco
from typing import Tuple
from mjlab.core.editors import SpecEditor
import abc

from mjlab import MJLAB_SRC_PATH, update_assets

_ASSETS_DIR = MJLAB_SRC_PATH / "entities" / "terrains" / "xmls" / "assets"


@dataclass(frozen=True)
class Skybox(SpecEditor):
  rgb1: Tuple[float, float, float] = (0.3, 0.5, 0.7)
  rgb2: Tuple[float, float, float] = (0.1, 0.2, 0.3)
  width: int = 512
  height: int = 3072

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    spec.add_texture(
      type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
      rgb1=self.rgb1,
      rgb2=self.rgb2,
      width=self.width,
      height=self.height,
    )


class TerrainEditor(SpecEditor, abc.ABC):
  @property
  @abc.abstractmethod
  def name(self) -> str:
    pass


@dataclass(frozen=True)
class FlatTerrain(TerrainEditor):
  name: str = "floor"
  size: tuple[float, float, float] = (10.0, 10.0, 0.1)
  rgba: tuple[float, float, float, float] = (0.4, 0.4, 0.4, 1.0)

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    spec.add_texture(
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
    spec.add_material(
      name="groundplane",
      texuniform=True,
      texrepeat=(4, 4),
      reflectance=0.2,
    ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

    spec.worldbody.add_geom(
      name=self.name,
      size=(0, 0, 0.01),
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      material="groundplane",
    )


@dataclass(frozen=True)
class RoughTerrain(TerrainEditor):
  name: str = "floor"

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    update_assets(spec.assets, _ASSETS_DIR)

    spec.add_hfield(
      name="hfield",
      file="hfield.png",
      size=(10, 10, 0.05, 0.1),
    )
    spec.add_texture(
      name="groundplane",
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      file="rocky_texture.png",
      colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR,
    )
    spec.add_material(
      name="groundplane",
      texuniform=True,
      texrepeat=(5, 5),
    ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

    spec.worldbody.add_geom(
      name="floor",
      size=(0, 0, 0.01),
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname="hfield",
      material="groundplane",
    )
