from dataclasses import dataclass
import mujoco

from mjlab.utils.spec_editor.spec_editor_config import (
  ActuatorCfg,
  TextureCfg,
  MaterialCfg,
  CollisionCfg,
  OptionCfg,
  GeomCfg,
  SensorCfg,
  LightCfg,
  CameraCfg,
)
from mjlab.utils.spec_editor.spec_editor_base import SpecEditor
from mjlab.utils.string import filter_exp, resolve_field
from mjlab.utils.spec import disable_collision, set_array_field, get_non_root_joints


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
    if self.cfg.body == "world":
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


@dataclass
class ActuatorEditor(SpecEditor):
  cfgs: tuple[ActuatorCfg, ...]

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    jnts = get_non_root_joints(spec)
    joint_names = [j.name for j in jnts]

    # Build a list of (cfg, joint_name) by resolving the config regex.
    flat: list[tuple[ActuatorCfg, str]] = []
    for cfg in self.cfgs:
      matched = filter_exp(cfg.joint_names_expr, joint_names)
      for name in matched:
        flat.append((cfg, name))

    # Sort by the joint index in the spec.
    flat.sort(key=lambda pair: joint_names.index(pair[1]))

    for cfg, jn in flat:
      spec.joint(jn).armature = cfg.armature
      spec.joint(jn).frictionloss = cfg.frictionloss

      act = spec.add_actuator(
        name=jn,
        target=jn,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        inheritrange=1.0,
        forcerange=(-cfg.effort_limit, cfg.effort_limit),
      )
      act.gainprm[0] = cfg.stiffness
      act.biasprm[1] = -cfg.stiffness
      act.biasprm[2] = -cfg.damping


@dataclass
class SensorEditor(SpecEditor):
  cfg: SensorCfg

  SENSOR_TYPE_MAP = {
    "gyro": mujoco.mjtSensor.mjSENS_GYRO,
    "upvector": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
    "velocimeter": mujoco.mjtSensor.mjSENS_VELOCIMETER,
    "framequat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
    "framepos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
    "framelinvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
    "frameangvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
    "framezaxis": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
    "accelerometer": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
  }

  SENSOR_OBJECT_TYPE_MAP = {
    "site": mujoco.mjtObj.mjOBJ_SITE,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "body": mujoco.mjtObj.mjOBJ_BODY,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    spec.add_sensor(
      name=self.cfg.name,
      type=self.SENSOR_TYPE_MAP[self.cfg.sensor_type],
      objtype=self.SENSOR_OBJECT_TYPE_MAP[self.cfg.object_type],
      objname=self.cfg.object_name,
    )


CAM_LIGHT_MODE_MAP = {
  "fixed": mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
  "track": mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
  "trackcom": mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
  "targetbody": mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
  "targetbodycom": mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM,
}


@dataclass
class CameraEditor(SpecEditor):
  cfg: CameraCfg

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    if self.cfg.body == "world":
      body = spec.worldbody
    else:
      body = spec.body(self.cfg.body)
    camera = body.add_camera(
      mode=CAM_LIGHT_MODE_MAP[self.cfg.mode],
      fovy=self.cfg.fovy,
      pos=self.cfg.pos,
      quat=self.cfg.quat,
    )
    if self.cfg.name is not None:
      camera.name = self.cfg.name
    if self.cfg.target is not None:
      camera.targetbody = self.cfg.target


@dataclass
class LightEditor(SpecEditor):
  cfg: LightCfg

  TYPE_MAP = {
    "directional": mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    "spot": mujoco.mjtLightType.mjLIGHT_SPOT,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    if self.cfg.body == "world":
      body = spec.worldbody
    else:
      body = spec.body(self.cfg.body)
    light = body.add_light(
      mode=CAM_LIGHT_MODE_MAP[self.cfg.mode],
      type=self.TYPE_MAP[self.cfg.type],
      castshadow=self.cfg.castshadow,
      pos=self.cfg.pos,
      dir=self.cfg.dir,
      cutoff=self.cfg.cutoff,
      exponent=self.cfg.exponent,
    )
    if self.cfg.name is not None:
      light.name = self.cfg.name
    if self.cfg.target is not None:
      light.targetbody = self.cfg.target
