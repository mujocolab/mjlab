from dataclasses import dataclass
import mujoco

from mjlab.core.editors import SpecEditor
from mjlab.entities.common.config import (
  TextureCfg,
  MaterialCfg,
  CollisionCfg,
  OptionCfg,
)
from mjlab.utils.string import filter_exp, resolve_field
from mjlab.utils.spec import disable_collision, set_array_field


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
class CollisionEditor(SpecEditor):
  cfg: CollisionCfg

  FIELD_DEFAULTS = {
    "condim": 1,
    "contype": 1,
    "conaffinity": 1,
    "priority": 0,
    "friction": None,
    "solref": None,
    "solimp": None,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    all_geoms: list[mujoco.MjsGeom] = spec.geoms
    all_geom_names = [g.name for g in all_geoms]
    geom_subset = filter_exp(self.cfg.geom_names_expr, all_geom_names)

    resolved_fields = {
      name: resolve_field(getattr(self.cfg, name), geom_subset, default)
      for name, default in self.FIELD_DEFAULTS.items()
    }

    for i, geom_name in enumerate(geom_subset):
      geom = spec.geom(geom_name)

      geom.condim = resolved_fields["condim"][i]
      geom.contype = resolved_fields["contype"][i]
      geom.conaffinity = resolved_fields["conaffinity"][i]
      geom.priority = resolved_fields["priority"][i]

      set_array_field(geom.friction, resolved_fields["friction"][i])
      set_array_field(geom.solref, resolved_fields["solref"][i])
      set_array_field(geom.solimp, resolved_fields["solimp"][i])

    if self.cfg.disable_other_geoms:
      other_geoms = set(all_geom_names).difference(geom_subset)
      for geom_name in other_geoms:
        geom = spec.geom(geom_name)
        disable_collision(geom)


@dataclass
class OptionEditor(SpecEditor):
  cfg: OptionCfg

  JACOBIAN_MAP = {
    "auto": mujoco.mjtJacobian.mjJAC_AUTO,
    "dense": mujoco.mjtJacobian.mjJAC_DENSE,
    "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
  }
  CONE_MAP = {
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
  }
  INTEGRATOR_MAP = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
  }
  SOLVER_MAP = {
    "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    "cg": mujoco.mjtSolver.mjSOL_CG,
    "pgs": mujoco.mjtSolver.mjSOL_PGS,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    attrs = {
      "jacobian": self.JACOBIAN_MAP[self.cfg.jacobian],
      "cone": self.CONE_MAP[self.cfg.cone],
      "integrator": self.INTEGRATOR_MAP[self.cfg.integrator],
      "solver": self.SOLVER_MAP[self.cfg.solver],
      "timestep": self.cfg.timestep,
      "impratio": self.cfg.impratio,
      "gravity": self.cfg.gravity,
      "iterations": self.cfg.iterations,
      "tolerance": self.cfg.tolerance,
      "ls_iterations": self.cfg.ls_iterations,
      "ls_tolerance": self.cfg.ls_tolerance,
    }
    for k, v in attrs.items():
      setattr(spec.option, k, v)
