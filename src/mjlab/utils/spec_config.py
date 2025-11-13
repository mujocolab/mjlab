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

  geom_names_expr: tuple[str, ...]
  """Tuple of regex patterns to match geom names."""
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
    all_geom_names = tuple(g.name for g in all_geoms)
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
