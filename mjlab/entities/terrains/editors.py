from dataclasses import dataclass
import mujoco

from mjlab.core.editors import SpecEditor
from mjlab.entities.terrains.terrain_config import MaterialCfg, TextureCfg, GeomCfg


@dataclass
class TextureEditor(SpecEditor):
  cfg: TextureCfg

  TYPE_MAP = {
    "2d": mujoco.mjtTexture.mjTEXTURE_2D,
    "cube": mujoco.mjtTexture.mjTEXTURE_CUBE,
    "skybox": mujoco.mjtTexture.mjTEXTURE_SKYBOX,
  }
  BUILIN_MAP = {
    "checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    "gradient": mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    "flat": mujoco.mjtBuiltin.mjBUILTIN_FLAT,
    "none": mujoco.mjtBuiltin.mjBUILTIN_NONE,
  }
  MARK_MAP = {
    "edge": mujoco.mjtMark.mjMARK_EDGE,
    "cross": mujoco.mjtMark.mjMARK_CROSS,
    "random": mujoco.mjtMark.mjMARK_RANDOM,
    "none": mujoco.mjtMark.mjMARK_NONE,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    spec.add_texture(
      name=self.cfg.name,
      type=self.TYPE_MAP[self.cfg.type],
      builtin=self.BUILIN_MAP[self.cfg.builtin],
      mark=self.MARK_MAP[self.cfg.mark],
      rgb1=self.cfg.rgb1,
      rgb2=self.cfg.rgb2,
      markrgb=self.cfg.markrgb,
      width=self.cfg.width,
      height=self.cfg.height,
    )


@dataclass
class MaterialEditor(SpecEditor):
  cfg: MaterialCfg

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    mat = spec.add_material(
      name=self.cfg.name,
      texuniform=self.cfg.texuniform,
      texrepeat=self.cfg.texrepeat,
    )
    if self.cfg.texture is not None:
      mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = self.cfg.texture


@dataclass
class GeomEditor(SpecEditor):
  cfg: GeomCfg

  TYPE_MAP = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "plane": mujoco.mjtGeom.mjGEOM_PLANE,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "hfield": mujoco.mjtGeom.mjGEOM_HFIELD,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    if self.cfg.body == "worldbody":
      body = spec.worldbody
    else:
      body = spec.body(self.cfg.body)
    geom = body.add_geom(
      name=self.cfg.name,
      type=self.TYPE_MAP[self.cfg.type],
      rgba=self.cfg.rgba,
      material=self.cfg.material,
      group=self.cfg.group,
    )
    for i in range(len(self.cfg.size)):
      geom.size[i] = self.cfg.size[i]
