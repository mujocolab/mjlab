from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco

from mjlab.sim import MujocoCfg
from mjlab.utils.spec import (
  disable_collision,
  get_non_free_joints,
  is_joint_limited,
)
from mjlab.utils.spec_editor.spec_editor_base import SpecEditor
from mjlab.utils.spec_editor.spec_editor_config import (
  ActuatorCfg,
  CameraCfg,
  CollisionCfg,
  ContactSensorCfg,
  GeomCfg,
  LightCfg,
  MaterialCfg,
  SensorCfg,
  TextureCfg,
)
from mjlab.utils.string import filter_exp, resolve_field


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
      mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = self.cfg.texture


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

  @staticmethod
  def set_array_field(field, values):
    if values is None:
      return
    for i, v in enumerate(values):
      field[i] = v

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

      CollisionEditor.set_array_field(geom.friction, resolved_fields["friction"][i])
      CollisionEditor.set_array_field(geom.solref, resolved_fields["solref"][i])
      CollisionEditor.set_array_field(geom.solimp, resolved_fields["solimp"][i])

    if self.cfg.disable_other_geoms:
      other_geoms = set(all_geom_names).difference(geom_subset)
      for geom_name in other_geoms:
        geom = spec.geom(geom_name)
        disable_collision(geom)


@dataclass
class OptionEditor(SpecEditor):
  cfg: MujocoCfg

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
  jnt_names: tuple[str, ...] | None = None

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    jnts = get_non_free_joints(spec)
    joint_names = [j.name for j in jnts]

    # Build a list of (cfg, joint_name) by resolving the config regex.
    flat: list[tuple[ActuatorCfg, str]] = []
    for cfg in self.cfgs:
      matched = filter_exp(cfg.joint_names_expr, joint_names)
      for name in matched:
        flat.append((cfg, name))

    # Sort by the joint index in the spec.
    flat.sort(key=lambda pair: joint_names.index(pair[1]))

    self.jnt_names = tuple(f[1] for f in flat)

    for cfg, jn in flat:
      spec.joint(jn).armature = cfg.armature
      spec.joint(jn).frictionloss = cfg.frictionloss

      if not is_joint_limited(spec.joint(jn)):
        raise ValueError(f"Joint {jn} is not limited.")

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
    "contact": mujoco.mjtSensor.mjSENS_CONTACT,
    "subtreeangmom": mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
  }

  SENSOR_OBJECT_TYPE_MAP = {
    "site": mujoco.mjtObj.mjOBJ_SITE,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "xbody": mujoco.mjtObj.mjOBJ_XBODY,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    sns = spec.add_sensor(
      name=self.cfg.name,
      type=self.SENSOR_TYPE_MAP[self.cfg.sensor_type],
      objtype=self.SENSOR_OBJECT_TYPE_MAP[self.cfg.objtype],
      objname=self.cfg.objname,
    )
    if self.cfg.reftype is not None and self.cfg.refname is not None:
      sns.reftype = self.SENSOR_OBJECT_TYPE_MAP[self.cfg.reftype]
      sns.refname = self.cfg.refname


@dataclass
class ContactSensorEditor(SpecEditor):
  cfg: ContactSensorCfg

  CONDATA_MAP = {
    "found": 0,
    "force": 1,
    "torque": 2,
    "dist": 3,
    "pos": 4,
    "normal": 5,
    "tangent": 6,
  }

  REDUCE_MAP = {
    "none": 0,
    "mindist": 1,
    "maxforce": 2,
    "netforce": 3,
  }

  @staticmethod
  def construct_contact_sensor_intprm(
    data: tuple[
      Literal["found", "force", "torque", "dist", "pos", "normal", "tangent"], ...
    ]
    | None,
    reduce: Literal["none", "mindist", "maxforce", "netforce"],
    num: int = 1,
  ) -> list[int]:
    if num <= 0:
      raise ValueError("'num' must be positive")
    if data:
      values = [ContactSensorEditor.CONDATA_MAP[k] for k in data]
      for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
          raise ValueError(
            "Data attributes must be in order: "
            f"{', '.join(ContactSensorEditor.CONDATA_MAP.keys())}"
          )
      dataspec = sum(1 << v for v in values)
    else:
      dataspec = 1
    return [dataspec, ContactSensorEditor.REDUCE_MAP.get(reduce, 0), num]

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    if self.cfg.geom1 is not None:
      objtype = mujoco.mjtObj.mjOBJ_GEOM
      objname = self.cfg.geom1
    elif self.cfg.body1 is not None:
      objtype = mujoco.mjtObj.mjOBJ_BODY
      objname = self.cfg.body1
    elif self.cfg.subtree1 is not None:
      objtype = mujoco.mjtObj.mjOBJ_XBODY
      objname = self.cfg.subtree1
    else:
      raise ValueError("One of geom1, body1, subtree1 must be non-None.")

    if self.cfg.geom2 is not None:
      reftype = mujoco.mjtObj.mjOBJ_GEOM
      refname = self.cfg.geom2
    elif self.cfg.body2 is not None:
      reftype = mujoco.mjtObj.mjOBJ_BODY
      refname = self.cfg.body2
    elif self.cfg.subtree2 is not None:
      reftype = mujoco.mjtObj.mjOBJ_XBODY
      refname = self.cfg.subtree2
    else:
      raise ValueError("One of geom2, body2, subtree2 must be non-None.")

    spec.add_sensor(
      name=self.cfg.name,
      type=mujoco.mjtSensor.mjSENS_CONTACT,
      objtype=objtype,
      objname=objname,
      reftype=reftype,
      refname=refname,
      intprm=self.construct_contact_sensor_intprm(
        data=self.cfg.data,
        reduce=self.cfg.reduce,
        num=self.cfg.num,
      ),
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
