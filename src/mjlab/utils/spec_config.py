import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import mujoco

from mjlab.utils.string import filter_exp, resolve_field

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

# Geom attributes that define the discrete contact structure: which pairs
# collide and what contact type they produce. CollisionCfg always writes these.
_GEOM_STRUCTURAL_FIELDS = ("contype", "conaffinity", "condim", "priority")
# Continuous contact parameters; None inherits the compiled XML value.
_GEOM_TUNING_FIELDS = ("friction", "solref", "solimp", "margin", "gap", "solmix")
_GEOM_COLLISION_FIELDS = _GEOM_STRUCTURAL_FIELDS + _GEOM_TUNING_FIELDS
# Fields written element-wise into a fixed-size MuJoCo array.
_GEOM_ARRAY_FIELDS = ("friction", "solref", "solimp")

_VALID_CONDIM = (1, 3, 4, 6)


def _field_items(value) -> list[tuple[str | None, Any]]:
  """Normalize a scalar-or-dict field to (pattern, value) pairs."""
  if isinstance(value, dict):
    return list(value.items())
  return [(None, value)]


def _validate_geom_attr(name: str, value) -> None:
  """Validate a scalar-or-dict geom attribute. None values are skipped."""
  for pattern, v in _field_items(value):
    if v is None or name in _GEOM_ARRAY_FIELDS:
      continue
    where = f" for pattern '{pattern}'" if pattern is not None else ""
    if name == "condim":
      if v not in _VALID_CONDIM:
        raise ValueError(f"condim must be one of {set(_VALID_CONDIM)}, got {v}{where}")
    elif name == "group":
      if not 0 <= v < mujoco.mjNGROUP:
        raise ValueError(f"group must be in [0, {mujoco.mjNGROUP}), got {v}{where}")
    elif name in ("contype", "conaffinity", "priority", "margin", "gap"):
      if v < 0:
        raise ValueError(f"{name} must be non-negative, got {v}{where}")
    elif name == "solmix":
      if not 0 <= v <= 1:
        raise ValueError(f"solmix must be in [0, 1], got {v}{where}")


def _set_geom_attr(geom: mujoco.MjsGeom, name: str, value) -> None:
  """Write a single geom attribute; None is a no-op."""
  if value is None:
    return
  if name in _GEOM_ARRAY_FIELDS:
    field = getattr(geom, name)
    for i, v in enumerate(value):
      field[i] = v
  else:
    setattr(geom, name, value)


def _matched_geoms(spec: mujoco.MjSpec, exprs: tuple[str, ...]) -> list[mujoco.MjsGeom]:
  """Return geom objects whose names match any expression.

  Filters the geom objects by name membership instead of looking names up via
  ``spec.geom(name)``, so unnamed geoms (which all share the empty-string name)
  are handled correctly. Geoms sharing a name are matched as a group.
  """
  all_geoms: list[mujoco.MjsGeom] = spec.geoms
  matched = set(filter_exp(exprs, tuple(g.name for g in all_geoms)))
  return [g for g in all_geoms if g.name in matched]


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
  builtin: Literal["checker", "gradient", "flat", "none"] = "none"
  """Built-in texture pattern ("checker", "gradient", "flat", or "none")."""
  rgb1: tuple[float, float, float] | None = None
  """First RGB color tuple."""
  rgb2: tuple[float, float, float] | None = None
  """Second RGB color tuple."""
  width: int | None = None
  """Texture width in pixels."""
  height: int | None = None
  """Texture height in pixels."""
  mark: Literal["edge", "cross", "random", "none"] = "none"
  """Marking pattern ("edge", "cross", "random", or "none")."""
  markrgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """RGB color for markings."""
  random: float = 0.01
  """Fraction of marked pixels when mark is "random"."""
  file: str | None = None
  """Path to an image file to load the texture from."""
  cubefiles: tuple[str, ...] | None = None
  """Paths to the six face images of a cube or skybox texture."""
  gridsize: tuple[int, int] | None = None
  """Rows and columns of the face grid in the file. Cube and skybox only."""
  gridlayout: str | None = None
  """Twelve-character face layout of the grid in the file. Cube and skybox only."""
  nchannel: int = 3
  """Number of channels in the texture."""
  hflip: bool = False
  """Whether to flip the loaded image horizontally."""
  vflip: bool = False
  """Whether to flip the loaded image vertically."""

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
      random=self.random,
      width=self.width,
      height=self.height,
      nchannel=self.nchannel,
      file=self.file,
      cubefiles=self.cubefiles,
      gridsize=self.gridsize,
      gridlayout=self.gridlayout,
      hflip=self.hflip,
      vflip=self.vflip,
    )

  def validate(self) -> None:
    if self.width is not None and self.width <= 0:
      raise ValueError("Texture width must be positive.")
    if self.height is not None and self.height <= 0:
      raise ValueError("Texture height must be positive.")
    if self.builtin == "none" and self.file is None and self.cubefiles is None:
      raise ValueError("Texture must specify a builtin pattern, a file, or cubefiles.")
    # A 1x1 grid compiles to a single image, but the Warp renderer samples skyboxes
    # as six faces stacked vertically. Any larger grid compiles to those six faces.
    # A builtin pattern takes precedence over the file and always fills six faces.
    if (
      self.type == "skybox"
      and self.builtin == "none"
      and self.file is not None
      and (self.gridsize is None or tuple(self.gridsize) == (1, 1))
    ):
      raise ValueError(
        "A skybox loaded from a single file must declare a gridsize larger than "
        "1x1, or be specified via cubefiles."
      )


@dataclass
class MaterialCfg(SpecCfg):
  """Configuration to add a material to the MuJoCo spec.

  Optionally assigns the material to geoms matching ``geom_names_expr``.
  """

  name: str
  """Name of the material."""
  rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
  """RGBA color of the material. Without a texture this is the direct
  surface color; with a texture it multiplies the texture colors."""
  texuniform: bool = False
  """Whether texture coordinates are uniform."""
  texrepeat: tuple[float, float] = (1.0, 1.0)
  """Texture repeat pattern (width, height). Must be positive."""
  reflectance: float = 0.0
  """Material reflectance value."""
  texture: str | None = None
  """Name of texture to apply (optional)."""
  geom_names_expr: tuple[str, ...] | None = None
  """Regex patterns to match geom names. Matching geoms will have their
  material set to this material. ``None`` means no assignment."""

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    mat = spec.add_material(
      name=self.name,
      rgba=self.rgba,
      texuniform=self.texuniform,
      texrepeat=self.texrepeat,
      reflectance=self.reflectance,
    )
    if self.texture is not None:
      mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = self.texture

    if self.geom_names_expr is not None:
      all_geom_names = tuple(g.name for g in spec.geoms)
      matched = filter_exp(self.geom_names_expr, all_geom_names)
      for geom_name in matched:
        spec.geom(geom_name).material = self.name

  def validate(self) -> None:
    if self.texrepeat[0] <= 0 or self.texrepeat[1] <= 0:
      raise ValueError("Material texrepeat values must be positive.")


@dataclass
class MeshCfg(SpecCfg):
  """Configuration to edit attributes of existing mesh assets in the MuJoCo spec.

  Unlike geom attributes, which are edited per geom via GeomCfg and CollisionCfg,
  the attributes here belong to the mesh asset itself. A mesh asset is a shared
  resource: one asset may be referenced by many
  mesh geoms across different bodies, and there is a single convex hull per asset.
  Attributes are therefore matched by mesh-asset name, not geom name.

  Only attributes set to a non-None value are applied; everything else is left at
  whatever the source XML compiled to.
  """

  mesh_names_expr: tuple[str, ...]
  """Regex patterns to match mesh-asset names."""
  maxhullvert: int | dict[str, int] | None = None
  """Maximum vertex count for each matched mesh's collision convex hull. Lower
  values yield cheaper narrowphase collision (the support function walks fewer hull
  vertices) at the cost of hull fidelity. -1 means unlimited (MuJoCo's default);
  positive values must be greater than 3. May be a single value applied to all
  matched meshes, or a dict mapping regex patterns to per-mesh values."""

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    all_mesh_names = tuple(m.name for m in spec.meshes)
    matched = filter_exp(self.mesh_names_expr, all_mesh_names)

    maxhullvert = resolve_field(self.maxhullvert, matched)
    for i, mesh_name in enumerate(matched):
      if maxhullvert[i] is not None:
        spec.mesh(mesh_name).maxhullvert = maxhullvert[i]

  def validate(self) -> None:
    if isinstance(self.maxhullvert, dict):
      items = list(self.maxhullvert.items())
    else:
      items = [(None, self.maxhullvert)]
    for pattern, value in items:
      if value is None or value == -1:
        continue
      if value <= 3:
        where = f" for pattern '{pattern}'" if pattern is not None else ""
        raise ValueError(
          f"maxhullvert must be -1 (unlimited) or greater than 3, got {value}{where}"
        )


@dataclass
class GeomCfg(SpecCfg):
  """Sparse patch on attributes of existing geoms in the MuJoCo spec.

  Geoms are matched by regex against their names; unnamed geoms match the empty
  string. Every attribute defaults to None, and only attributes set to a
  non-None value are applied; everything else is left at whatever the source
  XML compiled to.

  This is the single low-level editor for geom attributes. CollisionCfg is a
  higher-level policy built on the same machinery that *replaces* an entity's
  collision structure instead of patching it; prefer CollisionCfg for defining
  which geoms collide, and GeomCfg for targeted tweaks. Within an EntityCfg,
  CollisionCfg is applied after GeomCfg and overwrites any collision
  attributes it also touches.

  All attributes may be a single value applied to all matched geoms, or a dict
  mapping regex patterns to per-geom values (first matching pattern wins;
  geoms matching no pattern are left untouched).
  """

  geom_names_expr: tuple[str, ...]
  """Regex patterns to match geom names."""
  group: int | dict[str, int] | None = None
  """Visualization group for each matched geom. Viewers and sensors draw groups
  0-2 by default, so moving a geom to group 3-5 keeps it colliding while hiding
  it from rendering. Must be in [0, mjNGROUP): MuJoCo does not range-check the
  group, and out-of-range values are clamped into range by raycasting but
  dropped by the warp renderer."""
  contype: int | dict[str, int] | None = None
  """Collision type bitmask. Must be non-negative."""
  conaffinity: int | dict[str, int] | None = None
  """Collision affinity bitmask. Must be non-negative."""
  condim: int | dict[str, int] | None = None
  """Contact dimension. Must be one of {1, 3, 4, 6}."""
  priority: int | dict[str, int] | None = None
  """Collision priority. Must be non-negative."""
  friction: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  """Friction coefficients (sliding, torsional, rolling)."""
  solref: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  """Solver reference parameters."""
  solimp: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  """Solver impedance parameters."""
  margin: float | dict[str, float] | None = None
  """Detection margin. Contacts are generated when geom distance < margin."""
  gap: float | dict[str, float] | None = None
  """Gap for solver inclusion. Contact included when dist < margin - gap."""
  solmix: float | dict[str, float] | None = None
  """Mixing weight for blending solver parameters between geom pairs."""

  _PATCH_FIELDS = ("group",) + _GEOM_COLLISION_FIELDS

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    geom_subset = _matched_geoms(spec, self.geom_names_expr)
    subset_names = tuple(g.name for g in geom_subset)
    for field_name in self._PATCH_FIELDS:
      values = resolve_field(getattr(self, field_name), subset_names)
      for geom, value in zip(geom_subset, values, strict=True):
        _set_geom_attr(geom, field_name, value)

  def validate(self) -> None:
    for field_name in self._PATCH_FIELDS:
      _validate_geom_attr(field_name, getattr(self, field_name))


@dataclass
class CollisionCfg(SpecCfg):
  """Collision policy for the geoms in a MuJoCo spec.

  Unlike GeomCfg, which patches only the attributes you set, this config
  *replaces* the collision structure of every matched geom: ``contype``,
  ``conaffinity``, ``condim``, and ``priority`` are required and always
  written, so the contact behavior of matched geoms is fully determined by
  this config regardless of what the source XML compiled to. When one of
  these structural fields is a dict, its patterns must cover every matched
  geom; add a catch-all ``".*"`` entry for the default case.

  Continuous contact parameters (``friction``, ``solref``, ``solimp``,
  ``margin``, ``gap``, ``solmix``) are optional patches: None inherits the
  compiled XML value.

  With ``disable_other_geoms=True`` (the default), every geom *not* matched
  by ``geom_names_expr`` has its contype and conaffinity zeroed, so a single
  CollisionCfg describes the entire collision story of the spec it edits.
  """

  geom_names_expr: tuple[str, ...]
  """Tuple of regex patterns to match geom names."""
  contype: int | dict[str, int]
  """Collision type bitmask. Must be non-negative."""
  conaffinity: int | dict[str, int]
  """Collision affinity bitmask. Must be non-negative."""
  condim: int | dict[str, int]
  """Contact dimension. Must be one of {1, 3, 4, 6}."""
  priority: int | dict[str, int]
  """Collision priority. Must be non-negative."""
  friction: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  """Friction coefficients as tuple or dict mapping patterns to tuples."""
  solref: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  """Solver reference parameters as tuple or dict mapping patterns to tuples."""
  solimp: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
  """Solver impedance parameters as tuple or dict mapping patterns to tuples."""
  margin: float | dict[str, float] | None = None
  """Detection margin. Contacts are generated when geom distance < margin."""
  gap: float | dict[str, float] | None = None
  """Gap for solver inclusion. Contact included when dist < margin - gap."""
  solmix: float | dict[str, float] | None = None
  """Mixing weight for blending solver parameters between geom pairs."""
  disable_other_geoms: bool = True
  """Whether to disable collision for non-matching geoms."""

  def validate(self) -> None:
    for field_name in _GEOM_STRUCTURAL_FIELDS:
      if getattr(self, field_name) is None:
        raise ValueError(
          f"{field_name} is required: CollisionCfg fully determines the "
          "collision structure of matched geoms and cannot leave it to the "
          "source XML. Use GeomCfg for sparse patches."
        )
    for field_name in _GEOM_COLLISION_FIELDS:
      _validate_geom_attr(field_name, getattr(self, field_name))

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()

    all_geoms: list[mujoco.MjsGeom] = spec.geoms
    matched = set(filter_exp(self.geom_names_expr, tuple(g.name for g in all_geoms)))
    geom_subset = [g for g in all_geoms if g.name in matched]
    subset_names = tuple(g.name for g in geom_subset)

    for field_name in _GEOM_COLLISION_FIELDS:
      values = resolve_field(getattr(self, field_name), subset_names)
      structural = field_name in _GEOM_STRUCTURAL_FIELDS
      for geom, value in zip(geom_subset, values, strict=True):
        if structural and value is None:
          raise ValueError(
            f"{field_name} dict does not match geom '{geom.name}'. Structural "
            "collision fields must cover every matched geom; add a catch-all "
            "'.*' entry for the default case."
          )
        _set_geom_attr(geom, field_name, value)

    if self.disable_other_geoms:
      for geom in all_geoms:
        if geom.name not in matched:
          geom.contype = 0
          geom.conaffinity = 0


def warn_overlapping_geom_edits(
  geom_cfgs: Sequence[GeomCfg],
  collision_cfgs: Sequence[CollisionCfg],
  spec: mujoco.MjSpec,
) -> None:
  """Warn when GeomCfg collision patches would be overwritten by a CollisionCfg.

  Within an EntityCfg, collision configs are applied after geom configs, so any
  collision attribute both of them write on the same geom ends up with the
  CollisionCfg value. This lint makes that silent last-writer-wins visible.
  """
  if not geom_cfgs or not collision_cfgs:
    return

  all_names = tuple(g.name for g in spec.geoms)

  # (geom name, field) pairs each CollisionCfg will write.
  policy_writes: set[tuple[str, str]] = set()
  for ccfg in collision_cfgs:
    matched = set(filter_exp(ccfg.geom_names_expr, all_names))
    subset = tuple(sorted(matched))
    for field_name in _GEOM_COLLISION_FIELDS:
      values = resolve_field(getattr(ccfg, field_name), subset)
      structural = field_name in _GEOM_STRUCTURAL_FIELDS
      for name, value in zip(subset, values, strict=True):
        if structural or value is not None:
          policy_writes.add((name, field_name))
    if ccfg.disable_other_geoms:
      for name in set(all_names) - matched:
        policy_writes.add((name, "contype"))
        policy_writes.add((name, "conaffinity"))

  clobbered: set[str] = set()
  for gcfg in geom_cfgs:
    matched = set(filter_exp(gcfg.geom_names_expr, all_names))
    subset = tuple(sorted(matched))
    for field_name in _GEOM_COLLISION_FIELDS:
      values = resolve_field(getattr(gcfg, field_name), subset)
      for name, value in zip(subset, values, strict=True):
        if value is not None and (name, field_name) in policy_writes:
          clobbered.add(f"{name}.{field_name}")

  if clobbered:
    shown = ", ".join(sorted(clobbered)[:8])
    more = f" (and {len(clobbered) - 8} more)" if len(clobbered) > 8 else ""
    warnings.warn(
      f"GeomCfg sets collision attributes that a CollisionCfg (applied "
      f"after) overwrites: {shown}{more}. Set these through CollisionCfg "
      "instead.",
      stacklevel=3,
    )


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
  pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Light position (x, y, z)."""
  dir: tuple[float, float, float] = (0.0, 0.0, -1.0)
  """Light direction vector (x, y, z)."""
  cutoff: float = 45.0
  """Spot light cutoff angle in degrees."""
  exponent: float = 10.0
  """Spot light exponent."""
  diffuse: tuple[float, float, float] | None = None
  """Diffuse RGB color. ``None`` keeps MuJoCo's default."""
  specular: tuple[float, float, float] | None = None
  """Specular RGB color. ``None`` keeps MuJoCo's default."""
  ambient: tuple[float, float, float] | None = None
  """Ambient RGB color. ``None`` keeps MuJoCo's default."""
  active: bool = True
  """Whether the light is on."""
  attenuation: tuple[float, float, float] = (1.0, 0.0, 0.0)
  """Constant, linear, and quadratic attenuation coefficients."""

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
      diffuse=self.diffuse,
      specular=self.specular,
      ambient=self.ambient,
      active=self.active,
      attenuation=self.attenuation,
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
  fovy: float = 45.0
  """Field of view in degrees."""
  pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Camera position (x, y, z)."""
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
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
