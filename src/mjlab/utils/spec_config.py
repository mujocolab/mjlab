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
  "condim": 3,
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
  """Name of the texture."""
  type: Literal["2d", "cube", "skybox"]
  """Texture type ("2d", "cube", or "skybox")."""
  builtin: Literal["checker", "gradient", "flat", "none"]
  """Built-in texture pattern ("checker", "gradient", "flat", or "none")."""
  rgb1: tuple[float, float, float]
  """First RGB color tuple."""
  rgb2: tuple[float, float, float]
  """Second RGB color tuple."""
  width: int
  """Texture width in pixels (must be positive)."""
  height: int
  """Texture height in pixels (must be positive)."""
  mark: Literal["edge", "cross", "random", "none"] = "none"
  """Marking pattern ("edge", "cross", "random", or "none")."""
  markrgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """RGB color for markings."""

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
  """Name of the material."""
  texuniform: bool
  """Whether texture is uniform."""
  texrepeat: tuple[int, int]
  """Texture repeat pattern (width, height) - both must be positive."""
  reflectance: float = 0.0
  """Material reflectance value."""
  texture: str | None = None
  """Name of texture to apply (optional)."""

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
  """Configuration to modify collision properties of geoms in the MuJoCo spec.

  Supports regex pattern matching for geom names and dict-based field resolution
  for fine-grained control over collision properties.
  """

  geom_names_expr: list[str]
  """List of regex patterns to match geom names."""
  contype: int | dict[str, int] = 1
  """Collision type (int or dict mapping patterns to values). Must be non-negative."""
  conaffinity: int | dict[str, int] = 1
  """Collision affinity (int or dict mapping patterns to values). Must be
  non-negative."""
  condim: int | dict[str, int] = 3
  """Contact dimension (int or dict mapping patterns to values). Must be one
  of {1, 3, 4, 6}."""
  priority: int | dict[str, int] = 0
  """Collision priority (int or dict mapping patterns to values). Must be
  non-negative."""
  friction: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  """Friction coefficients as tuple or dict mapping patterns to tuples."""
  solref: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  """Solver reference parameters as tuple or dict mapping patterns to tuples."""
  solimp: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  """Solver impedance parameters as tuple or dict mapping patterns to tuples."""
  disable_other_geoms: bool = True
  """Whether to disable collision for non-matching geoms."""

  @staticmethod
  def set_array_field(field, values):
    if values is None:
      return
    for i, v in enumerate(values):
      field[i] = v

  def validate(self) -> None:
    """Validate collision configuration parameters."""
    valid_condim = {1, 3, 4, 6}

    # Validate condim specifically (has special valid values).
    if isinstance(self.condim, int):
      if self.condim not in valid_condim:
        raise ValueError(f"condim must be one of {valid_condim}, got {self.condim}")
    elif isinstance(self.condim, dict):
      for pattern, value in self.condim.items():
        if value not in valid_condim:
          raise ValueError(
            f"condim must be one of {valid_condim}, got {value} for pattern '{pattern}'"
          )

    # Validate other int parameters.
    if isinstance(self.contype, int) and self.contype < 0:
      raise ValueError("contype must be non-negative")
    if isinstance(self.conaffinity, int) and self.conaffinity < 0:
      raise ValueError("conaffinity must be non-negative")
    if isinstance(self.priority, int) and self.priority < 0:
      raise ValueError("priority must be non-negative")

    # Validate dict parameters (excluding condim which is handled above).
    for field_name in ["contype", "conaffinity", "priority"]:
      field_value = getattr(self, field_name)
      if isinstance(field_value, dict):
        for pattern, value in field_value.items():
          if value < 0:
            raise ValueError(
              f"{field_name} must be non-negative, got {value} for pattern '{pattern}'"
            )

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
  """Name of the light (optional)."""
  body: str = "world"
  """Body to attach light to (default: "world")."""
  mode: str = "fixed"
  """Light mode ("fixed", "track", "trackcom", "targetbody", "targetbodycom")."""
  target: str | None = None
  """Target body for tracking modes (optional)."""
  type: Literal["spot", "directional"] = "spot"
  """Light type ("spot" or "directional")."""
  castshadow: bool = True
  """Whether light casts shadows."""
  pos: tuple[float, float, float] = (0, 0, 0)
  """Light position (x, y, z)."""
  dir: tuple[float, float, float] = (0, 0, -1)
  """Light direction vector (x, y, z)."""
  cutoff: float = 45
  """Spot light cutoff angle in degrees."""
  exponent: float = 10
  """Spot light exponent."""

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
  """Name of the camera."""
  body: str = "world"
  """Body to attach camera to (default: "world")."""
  mode: str = "fixed"
  """Camera mode ("fixed", "track", "trackcom", "targetbody", "targetbodycom")."""
  target: str | None = None
  """Target body for tracking modes (optional)."""
  fovy: float = 45
  """Field of view in degrees."""
  pos: tuple[float, float, float] = (0, 0, 0)
  """Camera position (x, y, z)."""
  quat: tuple[float, float, float, float] = (1, 0, 0, 0)
  """Camera orientation quaternion (w, x, y, z)."""

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
  """Configuration for PD-controlled actuators applied to joints.

  Configures position-controlled actuators with PD control parameters,
  effort limits, and joint properties. Supports regex pattern matching
  for joint names.
  """

  joint_names_expr: list[str]
  """List of regex patterns to match joint names."""
  effort_limit: float
  """Maximum force/torque the actuator can apply (must be positive)."""
  stiffness: float
  """Position gain (P-gain) for PD control (must be non-negative)."""
  damping: float
  """Velocity gain (D-gain) for PD control (must be non-negative)."""
  frictionloss: float = 0.0
  """Joint friction loss coefficient (must be non-negative)."""
  armature: float = 0.0
  """Rotor inertia or reflected inertia for the joint (must be non-negative)."""


@dataclass
class ActuatorSetCfg(SpecCfg):
  """Configuration for a set of position-controlled actuators applied to joints.

  Applies multiple actuator configurations to joints matched by regex patterns.
  When multiple patterns match the same joint, the last matching configuration
  takes precedence. Actuators are created in the order joints appear in the spec
  to ensure deterministic behavior.
  """

  cfgs: tuple[ActuatorCfg, ...]
  """Tuple of ActuatorCfg instances to apply."""

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    from mjlab.utils.spec import get_non_free_joints, is_joint_limited
    from mjlab.utils.string import filter_exp

    self.validate()

    # Get all non-free joints in spec order.
    jnts = get_non_free_joints(spec)
    joint_names = [j.name for j in jnts]

    # Build list of (cfg, joint_name) by resolving each config's regex.
    cfg_joint_pairs: list[tuple[ActuatorCfg, str]] = []

    for cfg in self.cfgs:
      matched = filter_exp(cfg.joint_names_expr, joint_names)
      for joint_name in matched:
        cfg_joint_pairs.append((cfg, joint_name))

    # Check if any joints were matched (only if there are actuator configs).
    if self.cfgs and not cfg_joint_pairs:
      patterns = [f"'{expr}'" for cfg in self.cfgs for expr in cfg.joint_names_expr]
      available_joints = (
        [f"'{name}'" for name in joint_names] if joint_names else ["(none)"]
      )
      raise ValueError(
        f"No joints matched actuator patterns {', '.join(patterns)}. "
        f"Available joints: {', '.join(available_joints)}"
      )

    # Sort by joint order in spec (maintains deterministic ordering).
    cfg_joint_pairs.sort(key=lambda pair: joint_names.index(pair[1]))

    for cfg, joint_name in cfg_joint_pairs:
      joint = spec.joint(joint_name)

      if not is_joint_limited(joint):
        raise ValueError(f"Joint {joint_name} must be limited for position control")

      joint.armature = cfg.armature
      joint.frictionloss = cfg.frictionloss

      act = spec.add_actuator(
        name=joint_name,
        target=joint_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        inheritrange=1.0,
        forcerange=(-cfg.effort_limit, cfg.effort_limit),
      )

      act.gainprm[0] = cfg.stiffness
      act.biasprm[1] = -cfg.stiffness
      act.biasprm[2] = -cfg.damping

  def validate(self) -> None:
    """Validate all actuator configurations."""
    for cfg in self.cfgs:
      if cfg.effort_limit <= 0:
        raise ValueError(f"effort_limit must be positive, got {cfg.effort_limit}")
      if cfg.stiffness < 0:
        raise ValueError(f"stiffness must be non-negative, got {cfg.stiffness}")
      if cfg.damping < 0:
        raise ValueError(f"damping must be non-negative, got {cfg.damping}")
      if cfg.frictionloss < 0:
        raise ValueError(f"frictionloss must be non-negative, got {cfg.frictionloss}")
      if cfg.armature < 0:
        raise ValueError(f"armature must be non-negative, got {cfg.armature}")


@dataclass
class SensorCfg(SpecCfg):
  """Configuration to add a sensor to the MuJoCo spec."""

  name: str
  """Name of the sensor."""
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
  """Type of sensor."""
  objtype: Literal["xbody", "body", "geom", "site"]
  """Type of object to attach sensor to."""
  objname: str
  """Name of object to attach sensor to."""
  reftype: Literal["xbody", "body", "geom", "site"] | None = None
  """Reference object type for relative sensors (optional)."""
  refname: str | None = None
  """Name of reference object (optional)."""

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

  Ref: https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact
  """

  name: str
  """Name of the sensor."""

  # Primary object (exactly one must be specified).
  geom1: str | None = None
  """Primary geom name (exactly one primary object must be specified)."""
  body1: str | None = None
  """Primary body name (exactly one primary object must be specified)."""
  subtree1: str | None = None
  """Primary subtree name (exactly one primary object must be specified)."""
  site: str | None = None
  """Site volume (exactly one primary object must be specified, requires secondary object)."""

  # Secondary object (all optional).
  geom2: str | None = None
  """Secondary geom name (optional)."""
  body2: str | None = None
  """Secondary body name (optional)."""
  subtree2: str | None = None
  """Secondary subtree name (optional)."""
  num: int = 1
  """Max contacts to track (default: 1)."""

  data: tuple[
    Literal["found", "force", "torque", "dist", "pos", "normal", "tangent"], ...
  ] = ("found",)
  """Data to extract."""
  reduce: Literal["none", "mindist", "maxforce", "netforce"] = "none"
  """How to combine."""

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
