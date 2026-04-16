"""Contact sensors track collisions between geoms, bodies, or subtrees.

A ``ContactSensor`` resolves regex patterns into a set of *primary* MuJoCo
elements and (optionally) a single *secondary* element to filter against. Each
physics step it pulls per-primary contact data from MuJoCo's native contact
sensor and packages it into a batched ``ContactData`` dataclass.

Shape conventions on ``ContactData``:

- ``P`` = number of primary elements resolved from the pattern (see
  :attr:`ContactSensor.primary_names` for the index→name mapping).
- ``N`` = ``P * num_slots`` (per-contact axis, primary-major layout).
- Per-contact fields (``found``, ``force``, ``torque``, ``dist``, ``pos``,
  ``normal``, ``tangent``) have shape ``[B, N, ...]``.
- Per-primary fields (``current_air_time`` etc.) have shape ``[B, P]``.

Most users want the default ``num_slots=1``, in which case ``N == P`` and the
two shape families coincide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg

_CONTACT_DATA_MAP = {
  "found": 0,
  "force": 1,
  "torque": 2,
  "dist": 3,
  "pos": 4,
  "normal": 5,
  "tangent": 6,
}

_CONTACT_DATA_DIMS = {
  "found": 1,
  "force": 3,
  "torque": 3,
  "dist": 1,
  "pos": 3,
  "normal": 3,
  "tangent": 3,
}

_CONTACT_REDUCE_MAP = {
  "none": 0,
  "mindist": 1,
  "maxforce": 2,
  "netforce": 3,
}

_MODE_TO_OBJTYPE = {
  "geom": mujoco.mjtObj.mjOBJ_GEOM,
  "body": mujoco.mjtObj.mjOBJ_BODY,
  "subtree": mujoco.mjtObj.mjOBJ_XBODY,
}


@dataclass
class ContactMatch:
  """Specifies what to match on one side of a contact.

  Args:
    mode: MuJoCo element type to match against ("geom", "body", or "subtree").
    pattern: A regex (or tuple of regexes) matched against element names within
      ``entity``. If ``entity`` is unset, the pattern is treated as a literal
      MuJoCo name (no regex expansion).
    entity: Entity name to scope the pattern to. If ``None``/``""``, the
      pattern is taken as a literal MuJoCo name.
    exclude: Names to filter out of the match. Each entry is treated as a
      regex if it contains any regex metacharacters, otherwise as an exact
      name.
  """

  mode: Literal["geom", "body", "subtree"]
  pattern: str | tuple[str, ...]
  entity: str | None = None
  exclude: tuple[str, ...] = ()


@dataclass
class ContactSensorCfg(SensorCfg):
  """Configuration for a :class:`ContactSensor`.

  A contact sensor watches contacts between a ``primary`` set of elements and
  an optional ``secondary`` element. Patterns expand to multiple primaries
  (e.g. all four feet on a quadruped); each primary becomes one column on the
  per-contact axis of the output tensors.

  See the module docstring for shape conventions (``P``, ``N``, primary-major
  layout) and :attr:`ContactSensor.primary_names` for index→name lookup.

  Args:
    primary: Elements to measure (e.g. the robot's feet). Typically a regex
      that resolves to multiple elements.
    secondary: Optional filter on what the primary may contact (e.g. terrain).
      ``None`` means "any contact with a primary counts".
    fields: Which contact quantities to extract. Only requested fields are
      allocated and computed; the rest are ``None`` on ``ContactData``.

      - ``"found"``: 0 = no contact; >0 = number of matched contacts.
      - ``"force"``, ``"torque"``: 3D vectors in the contact frame (or in the
        global frame when ``reduce="netforce"`` or ``global_frame=True``).
      - ``"dist"``: penetration depth (scalar).
      - ``"pos"``, ``"normal"``, ``"tangent"``: 3D vectors in the global
        frame (normal points primary → secondary).

    reduce: How to collapse simultaneous contacts on the same primary down to
      ``num_slots`` representative contacts.

      - ``"none"``: fast, non-deterministic ordering.
      - ``"mindist"``: keep the closest (deepest) contacts.
      - ``"maxforce"``: keep the strongest contacts.
      - ``"netforce"``: sum all contacts into a single net wrench (always
        produces one slot per primary regardless of ``num_slots``).

    num_slots: Number of contacts to retain per primary after reduction.
      Almost always ``1``: most policies want one representative contact per
      primary, and pattern expansion already gives you many primaries. Only
      raise this when a single primary may have multiple distinct contacts
      you want to inspect separately, paired with ``reduce`` in
      ``{"none", "mindist", "maxforce"}``. Ignored by ``reduce="netforce"``.
    secondary_policy: How to handle a secondary pattern that resolves to
      multiple elements: ``"first"`` picks the first match, ``"any"`` drops
      the secondary filter entirely, ``"error"`` raises.
    track_air_time: Allocate per-primary air/contact time accumulators
      (useful for gait rewards). Requires ``"found"`` in ``fields``. Slot
      reduction within a primary uses an "any slot in contact" rule.
    global_frame: Rotate ``force``/``torque`` from the contact frame to the
      global frame. Requires ``"normal"`` and ``"tangent"`` in ``fields``.
      Implicit when ``reduce="netforce"``.
    history_length: If >0, keep a rolling buffer of the last N substeps of
      ``force``/``torque``/``dist`` data. Set to your decimation value so the
      buffer covers exactly one policy step; useful for catching brief
      collisions that resolve mid-substep. ``0`` disables the buffer.
    debug: Print each MuJoCo sensor as it is added to the spec. Useful for
      checking that pattern expansion produced the elements you expected.
  """

  primary: ContactMatch
  secondary: ContactMatch | None = None
  fields: tuple[str, ...] = ("found", "force")
  reduce: Literal["none", "mindist", "maxforce", "netforce"] = "maxforce"
  num_slots: int = 1
  secondary_policy: Literal["first", "any", "error"] = "first"
  track_air_time: bool = False
  global_frame: bool = False
  history_length: int = 0
  debug: bool = False

  def build(self) -> ContactSensor:
    return ContactSensor(self)


@dataclass
class _ContactSlot:
  """Maps one MuJoCo sensor (one primary, one field) to its sensordata view."""

  primary_name: str
  field_name: str
  sensor_name: str
  data_view: torch.Tensor | None = None


@dataclass
class _AirTimeState:
  """Tracks how long contacts have been in air/contact. Shape: [B, P]."""

  current_air_time: torch.Tensor
  last_air_time: torch.Tensor
  current_contact_time: torch.Tensor
  last_contact_time: torch.Tensor
  last_time: torch.Tensor


@dataclass
class ContactData:
  """Contact sensor output (only requested fields are populated).

  Shape conventions: P = number of primaries; N = P * num_slots (per-contact
  fields are laid out primary-major). Air-time fields are per-primary and
  reduce across slots (any slot in contact → primary in contact).
  """

  found: torch.Tensor | None = None
  """[B, N] 0=no contact, >0=match count"""
  force: torch.Tensor | None = None
  """[B, N, 3] contact frame (global if reduce="netforce" or global_frame=True)"""
  torque: torch.Tensor | None = None
  """[B, N, 3] contact frame (global if reduce="netforce" or global_frame=True)"""
  dist: torch.Tensor | None = None
  """[B, N] penetration depth"""
  pos: torch.Tensor | None = None
  """[B, N, 3] global frame"""
  normal: torch.Tensor | None = None
  """[B, N, 3] global frame, primary→secondary"""
  tangent: torch.Tensor | None = None
  """[B, N, 3] global frame"""

  current_air_time: torch.Tensor | None = None
  """[B, P] time in air per primary (if track_air_time=True)"""
  last_air_time: torch.Tensor | None = None
  """[B, P] duration of last air phase per primary (if track_air_time=True)"""
  current_contact_time: torch.Tensor | None = None
  """[B, P] time in contact per primary (if track_air_time=True)"""
  last_contact_time: torch.Tensor | None = None
  """[B, P] duration of last contact phase per primary (if track_air_time=True)"""

  force_history: torch.Tensor | None = None
  """[B, N, H, 3] contact forces over last H substeps (index 0 = most recent)"""
  torque_history: torch.Tensor | None = None
  """[B, N, H, 3] contact torques over last H substeps (index 0 = most recent)"""
  dist_history: torch.Tensor | None = None
  """[B, N, H] penetration depth over last H substeps (index 0 = most recent)"""


class ContactSensor(Sensor[ContactData]):
  """Tracks contacts with automatic pattern expansion to multiple MuJoCo sensors."""

  def __init__(self, cfg: ContactSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg

    if cfg.global_frame and cfg.reduce != "netforce":
      if "normal" not in cfg.fields or "tangent" not in cfg.fields:
        raise ValueError(
          f"Sensor '{cfg.name}': global_frame=True requires 'normal' and 'tangent' "
          "in fields (needed to build rotation matrix)"
        )

    self._slots: list[_ContactSlot] = []
    self._data: mjwarp.Data | None = None
    self._device: str | None = None
    self._air_time_state: _AirTimeState | None = None
    self._history_state: dict[str, torch.Tensor] | None = None

  @property
  def primary_names(self) -> list[str]:
    """Primary names in the order they appear along the per-contact dim.

    Per-contact fields ([B, N, ...]) are laid out primary-major, so primary
    `primary_names[i]` occupies indices [i * num_slots : (i + 1) * num_slots].
    Per-primary fields ([B, P, ...]) have one entry per name in this list.
    """
    return list(dict.fromkeys(slot.primary_name for slot in self._slots))

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    """Expand patterns and add MuJoCo sensors (one per primary x field pair)."""
    self._slots.clear()

    primary_names = self._resolve_primary_names(entities, self.cfg.primary)
    if self.cfg.secondary is None or self.cfg.secondary_policy == "any":
      secondary_name = None
    else:
      secondary_name = self._resolve_single_secondary(
        entities, self.cfg.secondary, self.cfg.secondary_policy
      )

    # MuJoCo allows packing multiple fields into one contact sensor via the
    # `dataspec` bitfield, but we register one sensor per (primary, field) pair
    # so each sensor's `sensordata` block is laid out as `[B, num_slots * dim]`
    # and `_extract_sensor_data` can reshape per field without computing
    # per-field offsets within an interleaved per-slot layout.
    for prim in primary_names:
      for field in self.cfg.fields:
        sensor_name = f"{self.cfg.name}_{prim}_{field}"

        self._add_contact_sensor_to_spec(
          scene_spec, sensor_name, prim, secondary_name, field
        )

        self._slots.append(
          _ContactSlot(
            primary_name=prim,
            field_name=field,
            sensor_name=sensor_name,
          )
        )

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    """Map sensors to sensordata buffer and allocate air time state."""
    del model

    if not self._slots:
      raise RuntimeError(
        f"There was an error initializing contact sensor '{self.cfg.name}'"
      )

    for slot in self._slots:
      sensor = mj_model.sensor(slot.sensor_name)
      start = sensor.adr[0]
      dim = sensor.dim[0]
      slot.data_view = data.sensordata[:, start : start + dim]

    self._data = data
    self._device = device

    n_primary = len(self.primary_names)

    if self.cfg.track_air_time:
      n_envs = data.time.shape[0]
      self._air_time_state = _AirTimeState(
        current_air_time=torch.zeros((n_envs, n_primary), device=device),
        last_air_time=torch.zeros((n_envs, n_primary), device=device),
        current_contact_time=torch.zeros((n_envs, n_primary), device=device),
        last_contact_time=torch.zeros((n_envs, n_primary), device=device),
        last_time=torch.zeros((n_envs,), device=device),
      )

    if self.cfg.history_length > 0:
      n_envs = data.time.shape[0]
      n_contacts = n_primary * self.cfg.num_slots
      h = self.cfg.history_length
      self._history_state = {}
      if "force" in self.cfg.fields:
        self._history_state["force"] = torch.zeros(
          (n_envs, n_contacts, h, 3), device=device
        )
      if "torque" in self.cfg.fields:
        self._history_state["torque"] = torch.zeros(
          (n_envs, n_contacts, h, 3), device=device
        )
      if "dist" in self.cfg.fields:
        self._history_state["dist"] = torch.zeros(
          (n_envs, n_contacts, h), device=device
        )

  def _compute_data(self) -> ContactData:
    out = self._extract_sensor_data()
    if self._air_time_state is not None:
      out.current_air_time = self._air_time_state.current_air_time
      out.last_air_time = self._air_time_state.last_air_time
      out.current_contact_time = self._air_time_state.current_contact_time
      out.last_contact_time = self._air_time_state.last_contact_time
    if self._history_state is not None:
      out.force_history = self._history_state.get("force")
      out.torque_history = self._history_state.get("torque")
      out.dist_history = self._history_state.get("dist")
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)

    # Reset air time state for specified envs.
    if self._air_time_state is not None:
      self._air_time_state.current_air_time[env_ids] = 0.0
      self._air_time_state.last_air_time[env_ids] = 0.0
      self._air_time_state.current_contact_time[env_ids] = 0.0
      self._air_time_state.last_contact_time[env_ids] = 0.0
      if self._data is not None:
        self._air_time_state.last_time[env_ids] = self._data.time[env_ids]

    # Reset history state for specified envs.
    if self._history_state is not None:
      for buf in self._history_state.values():
        buf[env_ids] = 0.0

  def update(self, dt: float) -> None:
    super().update(dt)
    if self._air_time_state is not None:
      self._update_air_time_tracking()
    if self._history_state is not None:
      self._update_history()

  def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
    """Returns [B, P] bool: True for primaries that landed within the last dt seconds."""
    if self._air_time_state is None:
      raise RuntimeError(
        f"Sensor '{self.cfg.name}' must have track_air_time=True "
        "to use compute_first_contact"
      )
    is_in_contact = self._air_time_state.current_contact_time > 0.0
    within_dt = self._air_time_state.current_contact_time < (dt + abs_tol)
    return is_in_contact & within_dt

  def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
    """Returns [B, P] bool: True for primaries that took off within the last dt seconds."""
    if self._air_time_state is None:
      raise RuntimeError(
        f"Sensor '{self.cfg.name}' must have track_air_time=True "
        "to use compute_first_air"
      )
    is_in_air = self._air_time_state.current_air_time > 0.0
    within_dt = self._air_time_state.current_air_time < (dt + abs_tol)
    return is_in_air & within_dt

  def _extract_sensor_data(self) -> ContactData:
    if not self._slots:
      raise RuntimeError(f"Sensor '{self.cfg.name}' not initialized")

    field_chunks: dict[str, list[torch.Tensor]] = {f: [] for f in self.cfg.fields}

    for slot in self._slots:
      assert slot.data_view is not None
      field_dim = _CONTACT_DATA_DIMS[slot.field_name]
      raw = slot.data_view.view(slot.data_view.size(0), self.cfg.num_slots, field_dim)
      field_chunks[slot.field_name].append(raw)

    out = ContactData()
    for field, chunks in field_chunks.items():
      cat = torch.cat(chunks, dim=1)
      if cat.size(-1) == 1:
        cat = cat.squeeze(-1)
      setattr(out, field, cat)

    if self.cfg.global_frame and self.cfg.reduce != "netforce":
      out = self._transform_to_global_frame(out)

    return out

  def _transform_to_global_frame(self, data: ContactData) -> ContactData:
    """Rotate force/torque from contact frame to global frame."""
    assert data.normal is not None and data.tangent is not None

    normal = data.normal
    tangent = data.tangent
    tangent2 = torch.cross(normal, tangent, dim=-1)
    R = torch.stack([tangent, tangent2, normal], dim=-1)

    has_contact = torch.norm(normal, dim=-1, keepdim=True) > 1e-8

    if data.force is not None:
      force_global = torch.einsum("...ij,...j->...i", R, data.force)
      data.force = torch.where(has_contact, force_global, data.force)

    if data.torque is not None:
      torque_global = torch.einsum("...ij,...j->...i", R, data.torque)
      data.torque = torch.where(has_contact, torque_global, data.torque)

    return data

  def _update_air_time_tracking(self) -> None:
    assert self._air_time_state is not None

    contact_data = self._extract_sensor_data()
    if contact_data.found is None or "found" not in self.cfg.fields:
      return

    assert self._data is not None
    current_time = self._data.time
    elapsed_time = current_time - self._air_time_state.last_time
    elapsed_time = elapsed_time.unsqueeze(-1)

    # Reduce `found` from [B, P*num_slots] to [B, P]: a primary is in contact
    # if any of its slots reports a match. Air-time is tracked per primary.
    found = contact_data.found
    if self.cfg.num_slots > 1:
      found = found.view(found.size(0), -1, self.cfg.num_slots).any(dim=-1)
    is_contact = found > 0

    state = self._air_time_state
    is_first_contact = (state.current_air_time > 0) & is_contact
    is_first_detached = (state.current_contact_time > 0) & ~is_contact

    state.last_air_time[:] = torch.where(
      is_first_contact,
      state.current_air_time + elapsed_time,
      state.last_air_time,
    )
    state.current_air_time[:] = torch.where(
      ~is_contact,
      state.current_air_time + elapsed_time,
      torch.zeros_like(state.current_air_time),
    )

    state.last_contact_time[:] = torch.where(
      is_first_detached,
      state.current_contact_time + elapsed_time,
      state.last_contact_time,
    )
    state.current_contact_time[:] = torch.where(
      is_contact,
      state.current_contact_time + elapsed_time,
      torch.zeros_like(state.current_contact_time),
    )

    state.last_time[:] = current_time

  def _update_history(self) -> None:
    """Roll history buffer and insert current contact data at index 0."""
    assert self._history_state is not None

    contact_data = self._extract_sensor_data()

    if "force" in self._history_state and contact_data.force is not None:
      self._history_state["force"] = self._history_state["force"].roll(1, dims=2)
      self._history_state["force"][:, :, 0, :] = contact_data.force

    if "torque" in self._history_state and contact_data.torque is not None:
      self._history_state["torque"] = self._history_state["torque"].roll(1, dims=2)
      self._history_state["torque"][:, :, 0, :] = contact_data.torque

    if "dist" in self._history_state and contact_data.dist is not None:
      self._history_state["dist"] = self._history_state["dist"].roll(1, dims=2)
      self._history_state["dist"][:, :, 0] = contact_data.dist

  def _resolve_primary_names(
    self, entities: dict[str, Entity], match: ContactMatch
  ) -> list[str]:
    if match.entity in (None, ""):
      result = (
        [match.pattern] if isinstance(match.pattern, str) else list(match.pattern)
      )
      return result

    if match.entity not in entities:
      raise ValueError(
        f"Primary entity '{match.entity}' not found. Available: {list(entities.keys())}"
      )
    ent = entities[match.entity]

    patterns = [match.pattern] if isinstance(match.pattern, str) else match.pattern

    if match.mode == "geom":
      _, names = ent.find_geoms(patterns)
    elif match.mode == "body":
      _, names = ent.find_bodies(patterns)
    elif match.mode == "subtree":
      _, names = ent.find_bodies(patterns)
      if not names:
        raise ValueError(
          f"Primary subtree pattern '{match.pattern}' matched no bodies in "
          f"'{match.entity}'"
        )
    else:
      raise ValueError("Primary mode must be one of {'geom','body','subtree'}")

    excludes = match.exclude
    if excludes:
      exclude_patterns = []
      exclude_exact = set()
      for exc in excludes:
        if any(c in exc for c in r".*+?[]{}()\|^$"):
          exclude_patterns.append(re.compile(exc))
        else:
          exclude_exact.add(exc)
      if exclude_exact:
        names = [n for n in names if n not in exclude_exact]
      if exclude_patterns:
        names = [n for n in names if not any(rx.search(n) for rx in exclude_patterns)]

    if not names:
      raise ValueError(
        f"Primary pattern '{match.pattern}' (after excludes) matched "
        f"no names in '{match.entity}'"
      )
    return names

  def _resolve_single_secondary(
    self,
    entities: dict[str, Entity],
    match: ContactMatch,
    policy: Literal["first", "any", "error"],
  ) -> str | None:
    if policy == "any":
      return None

    if isinstance(match.pattern, tuple):
      raise ValueError(
        "Secondary must specify a single name (string). "
        "Use a single exact name or a regex that resolves to one name, "
        "or set secondary_policy='any' if you want no filter."
      )

    if match.entity in (None, ""):
      if match.mode not in {"geom", "body", "subtree"}:
        raise ValueError("Secondary mode must be one of {'geom','body','subtree'}")
      return match.pattern

    if match.entity not in entities:
      raise ValueError(
        f"Secondary entity '{match.entity}' not found. "
        f"Available: {list(entities.keys())}"
      )
    ent = entities[match.entity]

    if match.mode == "subtree":
      return match.pattern

    if match.mode == "geom":
      _, names = ent.find_geoms(match.pattern)
    elif match.mode == "body":
      _, names = ent.find_bodies(match.pattern)
    else:
      raise ValueError("Secondary mode must be one of {'geom','body','subtree'}")

    if not names:
      raise ValueError(
        f"Secondary pattern '{match.pattern}' matched nothing in '{match.entity}'"
      )

    if len(names) == 1 or policy == "first":
      return names[0]

    raise ValueError(
      f"Secondary pattern '{match.pattern}' matched multiple: {names}. "
      f"Be explicit or set secondary_policy='first' or 'any'."
    )

  def _add_contact_sensor_to_spec(
    self,
    scene_spec: mujoco.MjSpec,
    sensor_name: str,
    primary_name: str,
    secondary_name: str | None,
    field: str,
  ) -> None:
    data_bits = 1 << _CONTACT_DATA_MAP[field]
    reduce_mode = _CONTACT_REDUCE_MAP[self.cfg.reduce]
    intprm = [data_bits, reduce_mode, self.cfg.num_slots]

    kwargs: dict[str, Any] = {
      "name": sensor_name,
      "type": mujoco.mjtSensor.mjSENS_CONTACT,
      "objtype": _MODE_TO_OBJTYPE[self.cfg.primary.mode],
      "objname": _prefix_name(primary_name, self.cfg.primary.entity),
      "intprm": intprm,
    }

    if secondary_name is not None:
      assert self.cfg.secondary is not None
      kwargs["reftype"] = _MODE_TO_OBJTYPE[self.cfg.secondary.mode]
      kwargs["refname"] = _prefix_name(secondary_name, self.cfg.secondary.entity)

    if self.cfg.debug:
      self._print_debug(sensor_name, field, intprm, kwargs)

    scene_spec.add_sensor(**kwargs)

  def _print_debug(
    self,
    sensor_name: str,
    field: str,
    intprm: list[int],
    kwargs: dict[str, Any],
  ) -> None:
    objtype_name = _objtype_name(kwargs["objtype"])
    reftype_val = kwargs.get("reftype")
    refname_val = kwargs.get("refname")
    if refname_val is None:
      ref_str = "<any>"
    else:
      ref_str = f"{_objtype_name(reftype_val)}:{refname_val}"
    print(
      "Adding contact sensor\n"
      f"  name    : {sensor_name}\n"
      f"  object  : {objtype_name}:{kwargs['objname']}\n"
      f"  ref     : {ref_str}\n"
      f"  field   : {field}  bits=0b{intprm[0]:b}\n"
      f"  reduce  : {self.cfg.reduce}  num_slots={self.cfg.num_slots}"
    )


def _prefix_name(name: str, entity: str | None) -> str:
  """Prepend ``entity/`` to a MuJoCo name when an entity scope is set."""
  if entity:
    return f"{entity}/{name}"
  return name


def _objtype_name(objtype: Any) -> str:
  """Pretty-print a MuJoCo object type, dropping the ``mjOBJ_`` prefix."""
  return getattr(objtype, "name", str(objtype)).removeprefix("mjOBJ_")
