import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import mujoco

_TYPE_MAP = {
  "2d": mujoco.mjtTexture.mjTEXTURE_2D,
  "cube": mujoco.mjtTexture.mjTEXTURE_CUBE,
  "skybox": mujoco.mjtTexture.mjTEXTURE_SKYBOX,
}
_BUILTIN_MAP = {
  "checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
  "gradient": mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
  "flat": mujoco.mjtBuiltin.mjBUILTIN_FLAT,
  "none": mujoco.mjtBuiltin.mjBUILTIN_NONE,
}
_MARK_MAP = {
  "edge": mujoco.mjtMark.mjMARK_EDGE,
  "cross": mujoco.mjtMark.mjMARK_CROSS,
  "random": mujoco.mjtMark.mjMARK_RANDOM,
  "none": mujoco.mjtMark.mjMARK_NONE,
}

_GEOM_ATTR_DEFAULTS = {
  "condim": 1,
  "contype": 1,
  "conaffinity": 1,
  "priority": 0,
  "friction": None,
  "solref": None,
  "solimp": None,
}

_LIGHT_TYPE_MAP = {
  "directional": mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
  "spot": mujoco.mjtLightType.mjLIGHT_SPOT,
}

_CAM_LIGHT_MODE_MAP = {
  "fixed": mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
  "track": mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
  "trackcom": mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
  "targetbody": mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
  "targetbodycom": mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM,
}

_SENSOR_TYPE_MAP = {
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

_SENSOR_OBJECT_TYPE_MAP = {
  "site": mujoco.mjtObj.mjOBJ_SITE,
  "geom": mujoco.mjtObj.mjOBJ_GEOM,
  "body": mujoco.mjtObj.mjOBJ_BODY,
  "xbody": mujoco.mjtObj.mjOBJ_XBODY,
}


_CONTACT_DATA_MAP = {
  "found": 0,
  "force": 1,
  "torque": 2,
  "dist": 3,
  "pos": 4,
  "normal": 5,
  "tangent": 6,
}

_CONTACT_REDUCE_MAP = {
  "none": 0,
  "mindist": 1,
  "maxforce": 2,
  "netforce": 3,
}


@dataclass
class SpecCfg(ABC):
  """Base class for all MuJoCo spec configurations."""

  @abstractmethod
  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    raise NotImplementedError

  def validate(self) -> None:  # noqa: B027
    """Optional validation method to be overridden by subclasses."""
    pass


@dataclass
class TextureCfg(SpecCfg):
  """Configuration to add a texture to the MuJoCo spec."""

  name: str
  type: Literal["2d", "cube", "skybox"]
  builtin: Literal["checker", "gradient", "flat", "none"]
  rgb1: tuple[float, float, float]
  rgb2: tuple[float, float, float]
  width: int
  height: int
  mark: Literal["edge", "cross", "random", "none"] = "none"
  markrgb: tuple[float, float, float] = (0.0, 0.0, 0.0)

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    spec.add_texture(
      name=self.name,
      type=_TYPE_MAP[self.type],
      builtin=_BUILTIN_MAP[self.builtin],
      mark=_MARK_MAP[self.mark],
      rgb1=self.rgb1,
      rgb2=self.rgb2,
      markrgb=self.markrgb,
      width=self.width,
      height=self.height,
    )

  def validate(self) -> None:
    if self.width <= 0 or self.height <= 0:
      raise ValueError("Texture width and height must be positive.")


@dataclass
class MaterialCfg(SpecCfg):
  """Configuration to add a material to the MuJoCo spec."""

  name: str
  texuniform: bool
  texrepeat: tuple[int, int]
  reflectance: float = 0.0
  texture: str | None = None

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    mat = spec.add_material(
      name=self.name,
      texuniform=self.texuniform,
      texrepeat=self.texrepeat,
    )
    if self.texture is not None:
      mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = self.texture

  def validate(self) -> None:
    if self.texrepeat[0] <= 0 or self.texrepeat[1] <= 0:
      raise ValueError("Material texrepeat values must be positive.")


@dataclass
class CollisionCfg(SpecCfg):
  """Configuration to modify collision properties of geoms in the MuJoCo spec."""

  geom_names_expr: list[str]
  contype: int | dict[str, int] = 1
  conaffinity: int | dict[str, int] = 1
  condim: int | dict[str, int] = 3
  priority: int | dict[str, int] = 0
  friction: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  solref: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  solimp: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  disable_other_geoms: bool = True

  @staticmethod
  def set_array_field(field, values):
    if values is None:
      return
    for i, v in enumerate(values):
      field[i] = v

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    from mjlab.utils.spec import disable_collision
    from mjlab.utils.string import filter_exp, resolve_field

    self.validate()

    all_geoms: list[mujoco.MjsGeom] = spec.geoms
    all_geom_names = [g.name for g in all_geoms]
    geom_subset = filter_exp(self.geom_names_expr, all_geom_names)

    resolved_fields = {
      name: resolve_field(getattr(self, name), geom_subset, default)
      for name, default in _GEOM_ATTR_DEFAULTS.items()
    }

    for i, geom_name in enumerate(geom_subset):
      geom = spec.geom(geom_name)

      geom.condim = resolved_fields["condim"][i]
      geom.contype = resolved_fields["contype"][i]
      geom.conaffinity = resolved_fields["conaffinity"][i]
      geom.priority = resolved_fields["priority"][i]

      CollisionCfg.set_array_field(geom.friction, resolved_fields["friction"][i])
      CollisionCfg.set_array_field(geom.solref, resolved_fields["solref"][i])
      CollisionCfg.set_array_field(geom.solimp, resolved_fields["solimp"][i])

    if self.disable_other_geoms:
      other_geoms = set(all_geom_names).difference(geom_subset)
      for geom_name in other_geoms:
        geom = spec.geom(geom_name)
        disable_collision(geom)


@dataclass
class LightCfg(SpecCfg):
  """Configuration to add a light to the MuJoCo spec."""

  name: str | None = None
  body: str = "world"
  mode: str = "fixed"
  target: str | None = None
  type: Literal["spot", "directional"] = "spot"
  castshadow: bool = True
  pos: tuple[float, float, float] = (0, 0, 0)
  dir: tuple[float, float, float] = (0, 0, -1)
  cutoff: float = 45
  exponent: float = 10

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    if self.body == "world":
      body = spec.worldbody
    else:
      body = spec.body(self.body)
    light = body.add_light(
      mode=_CAM_LIGHT_MODE_MAP[self.mode],
      type=_LIGHT_TYPE_MAP[self.type],
      castshadow=self.castshadow,
      pos=self.pos,
      dir=self.dir,
      cutoff=self.cutoff,
      exponent=self.exponent,
    )
    if self.name is not None:
      light.name = self.name
    if self.target is not None:
      light.targetbody = self.target


@dataclass
class CameraCfg(SpecCfg):
  """Configuration to add a camera to the MuJoCo spec."""

  name: str
  body: str = "world"
  mode: str = "fixed"
  target: str | None = None
  fovy: float = 45
  pos: tuple[float, float, float] = (0, 0, 0)
  quat: tuple[float, float, float, float] = (1, 0, 0, 0)

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    if self.body == "world":
      body = spec.worldbody
    else:
      body = spec.body(self.body)
    camera = body.add_camera(
      mode=_CAM_LIGHT_MODE_MAP[self.mode],
      fovy=self.fovy,
      pos=self.pos,
      quat=self.quat,
    )
    if self.name is not None:
      camera.name = self.name
    if self.target is not None:
      camera.targetbody = self.target


@dataclass
class ActuatorCfg:
  """Configuration for PD-controlled actuators applied to joints."""

  joint_names_expr: list[str]
  """List of regex patterns to match joint names."""
  effort_limit: dict[str, float] | float
  """Maximum force/torque the actuator can apply."""
  stiffness: dict[str, float] | float
  """Position gain (P-gain) for PD control."""
  damping: dict[str, float] | float
  """Velocity gain (D-gain) for PD control."""
  frictionloss: dict[str, float] | float = 0.0
  """Joint friction loss coefficient."""
  armature: dict[str, float] | float = 0.0
  """Rotor inertia or reflected inertia for the joint."""


@dataclass
class ActuatorSetCfg(SpecCfg):
  """Configuration for a set of position-controlled actuators applied to joints.

  Applies multiple actuator configurations to joints matched by regex patterns.
  When multiple patterns match the same joint, the last matching configuration
  takes precedence. Actuators are created in the order joints appear in the spec
  to ensure deterministic behavior.
  """

  cfgs: tuple[ActuatorCfg, ...]

  def _resolve_per_joints(self, joint_name, field) -> float:
    if isinstance(field, float):
      return field
    if joint_name in field:
      return field[joint_name]

    for j_expr, v in field.items():
      if re.fullmatch(j_expr, joint_name):
        return v

    raise ValueError(f"No value provided for joint: {joint_name}")

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    from mjlab.utils.spec import get_non_free_joints, is_joint_limited
    from mjlab.utils.string import filter_exp

    # Get all non-free joints in spec order.
    # TODO LOUIS: what if we don't want to actuate all joints...
    jnts = get_non_free_joints(spec)
    joint_names = [j.name for j in jnts]

    last_cfg_for_joint: dict[str, ActuatorCfg] = {}
    for cfg in self.cfgs:
      # TODO: check if already added and raise error instead of using last
      matched = filter_exp(cfg.joint_names_expr, joint_names)
      for jn in matched:
        last_cfg_for_joint[jn] = cfg

    for jn in joint_names:
      # non actuated joints
      cfg = last_cfg_for_joint.get(jn)
      if cfg is None:
        continue

      joint = spec.joint(jn)
      if not is_joint_limited(joint):
        raise ValueError(f"Joint {jn} must be limited for position control")

      joint_armature = self._resolve_per_joints(jn, last_cfg_for_joint[jn].armature)
      joint_frictionloss = self._resolve_per_joints(
        jn, last_cfg_for_joint[jn].frictionloss
      )
      joint_effort_limit = self._resolve_per_joints(
        jn, last_cfg_for_joint[jn].effort_limit
      )
      joint_stiffness = self._resolve_per_joints(jn, last_cfg_for_joint[jn].stiffness)
      joint_damping = self._resolve_per_joints(jn, last_cfg_for_joint[jn].damping)

      self._validate(
        jn,
        joint_armature,
        joint_frictionloss,
        joint_effort_limit,
        joint_stiffness,
        joint_damping,
      )

      joint.armature = joint_armature
      joint.frictionloss = joint_frictionloss

      act = spec.add_actuator(
        name=jn,
        target=jn,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        inheritrange=1.0,
        forcerange=(-joint_effort_limit, joint_effort_limit),
      )

      act.gainprm[0] = joint_stiffness
      act.biasprm[1] = -joint_stiffness
      act.biasprm[2] = -joint_damping

  def _validate(
    self,
    joint_name,
    joint_armature,
    joint_frictionloss,
    joint_effort_limit,
    joint_stiffness,
    joint_damping,
  ) -> None:
    """Validate joint configuration."""
    if joint_effort_limit <= 0:
      raise ValueError(
        f"effort_limit must be positive, got {joint_effort_limit} for {joint_name}"
      )
    if joint_stiffness < 0:
      raise ValueError(
        f"stiffness must be non-negative, got {joint_stiffness} for {joint_name}"
      )
    if joint_damping < 0:
      raise ValueError(
        f"damping must be non-negative, got {joint_damping} for {joint_name}"
      )
    if joint_frictionloss < 0:
      raise ValueError(
        f"frictionloss must be non-negative, got {joint_frictionloss} for {joint_name}"
      )
    if joint_armature < 0:
      raise ValueError(
        f"armature must be non-negative, got {joint_armature} for {joint_name}"
      )


@dataclass
class SensorCfg(SpecCfg):
  """Configuration to add a sensor to the MuJoCo spec."""

  name: str
  sensor_type: Literal[
    "gyro",
    "upvector",
    "velocimeter",
    "framequat",
    "framepos",
    "framelinvel",
    "frameangvel",
    "framezaxis",
    "accelerometer",
    "contact",
    "subtreeangmom",
  ]
  objtype: Literal["xbody", "body", "geom", "site"]
  objname: str
  reftype: Literal["xbody", "body", "geom", "site"] | None = None
  refname: str | None = None

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    sns = spec.add_sensor(
      name=self.name,
      type=_SENSOR_TYPE_MAP[self.sensor_type],
      objtype=_SENSOR_OBJECT_TYPE_MAP[self.objtype],
      objname=self.objname,
    )
    if self.reftype is not None and self.refname is not None:
      sns.reftype = _SENSOR_OBJECT_TYPE_MAP[self.reftype]
      sns.refname = self.refname


@dataclass
class ContactSensorCfg(SpecCfg):
  """Configuration for a contact sensor.

  Selects contacts from mjData.contact using intersection of specified criteria.
  Must specify a primary object (geom1/body1/subtree1/site), and optionally a
  secondary object (geom2/body2/subtree2) to match contacts between them.

  Examples:
    - ContactSensorCfg(name="hand_contacts", body1="hand")  # Any contact with hand
    - ContactSensorCfg(name="self_collisions", subtree1="arm", subtree2="arm")  # Arm self-collisions
    - ContactSensorCfg(name="table_contacts", geom1="table", body2="robot")  # Table-robot contacts
    - ContactSensorCfg(name="in_zone", site="zone1")  # Contacts within zone1 volume

  Ref: https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact
  """

  name: str

  # Primary object (exactly one must be specified).
  geom1: str | None = None
  body1: str | None = None
  subtree1: str | None = None
  site: str | None = None

  # Secondary object (all optional).
  geom2: str | None = None
  body2: str | None = None
  subtree2: str | None = None

  num: int = 1
  data: tuple[
    Literal["found", "force", "torque", "dist", "pos", "normal", "tangent"], ...
  ] = ("found",)
  reduce: Literal["none", "mindist", "maxforce", "netforce"] = "none"

  def _construct_intprm(self) -> list[int]:
    """Construct the intprm parameter for contact sensors."""
    if self.num <= 0:
      raise ValueError("'num' must be positive")

    if self.data:
      values = [_CONTACT_DATA_MAP[k] for k in self.data]
      for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
          raise ValueError(
            f"Data attributes must be in order: {', '.join(_CONTACT_DATA_MAP.keys())}"
          )
      dataspec = sum(1 << v for v in values)
    else:
      dataspec = 1

    return [dataspec, _CONTACT_REDUCE_MAP[self.reduce], self.num]

  def validate(self) -> None:
    # Exactly one primary object must be specified.
    group1_count = sum(
      x is not None for x in [self.geom1, self.body1, self.subtree1, self.site]
    )
    if group1_count != 1:
      raise ValueError(
        "Exactly one of geom1, body1, subtree1, or site must be specified"
      )

    # At most one secondary object.
    group2_count = sum(x is not None for x in [self.geom2, self.body2, self.subtree2])
    if group2_count > 1:
      raise ValueError("At most one of geom2, body2, subtree2 can be specified")

    # Site can only be used with group2 objects (not alone or with group1).
    if self.site is not None and group2_count == 0:
      raise ValueError(
        "Site must be used with a secondary object (geom2, body2, or subtree2)"
      )

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    # Determine primary object (exactly one will be set due to validation).
    if self.geom1 is not None:
      objtype = mujoco.mjtObj.mjOBJ_GEOM
      objname = self.geom1
    elif self.body1 is not None:
      objtype = mujoco.mjtObj.mjOBJ_BODY
      objname = self.body1
    elif self.subtree1 is not None:
      objtype = mujoco.mjtObj.mjOBJ_XBODY
      objname = self.subtree1
    else:  # self.site must be not None.
      objtype = mujoco.mjtObj.mjOBJ_SITE
      objname = self.site

    sensor_kwargs = {
      "name": self.name,
      "type": mujoco.mjtSensor.mjSENS_CONTACT,
      "objtype": objtype,
      "objname": objname,
      "intprm": self._construct_intprm(),
    }

    # Add secondary object if specified.
    if self.geom2 is not None:
      sensor_kwargs["reftype"] = mujoco.mjtObj.mjOBJ_GEOM
      sensor_kwargs["refname"] = self.geom2
    elif self.body2 is not None:
      sensor_kwargs["reftype"] = mujoco.mjtObj.mjOBJ_BODY
      sensor_kwargs["refname"] = self.body2
    elif self.subtree2 is not None:
      sensor_kwargs["reftype"] = mujoco.mjtObj.mjOBJ_XBODY
      sensor_kwargs["refname"] = self.subtree2

    spec.add_sensor(**sensor_kwargs)
