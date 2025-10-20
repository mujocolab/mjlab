"""Pattern-based contact sensor wrapper for MuJoCo.

MuJoCo Contact Sensor API
==========================

A MuJoCo contact sensor filters contacts from mjData.contact and reports a fixed-size
array. It uses intersection-based matching:

  - Specify criteria for BOTH sides: {geom1, body1, subtree1} AND {geom2, body2, subtree2}
  - Example: body1="left_foot" AND body2="terrain" → only foot-ground contacts
  - Subtree: body and all descendants (subtree1=subtree2=same body → self-collisions)
  - Empty criteria → matches all contacts

Processing stages:
  1. Matching: Select contacts using intersection criteria
  2. Reduction: Pick top N via reduce mode (none/mindist/maxforce/netforce)
  3. Extraction: Copy fields to output (found/force/torque/dist/pos/normal/tangent)

Output is fixed-size [num_slots x fields]. Empty slots filled with zeros.
Normal direction: points from "1" side to "2" side.

What This Wrapper Adds
======================

Pattern Expansion: Regex patterns → multiple MuJoCo sensors

  primary=ContactMatch(mode="geom", pattern=".*_foot", entity="robot")
  # Matches 4 feet → creates 4 MuJoCo sensors, one per foot
  # Each sensor tracks num_slots contacts → output shape [B, 4*num_slots, field_dim]

Structured Output: Named fields instead of raw arrays

  contact = scene.sensors["feet"].data
  forces = contact.force           # [B, N*num_slots, 3]
  in_contact = contact.found > 0   # [B, N*num_slots]

Air Time Tracking: Optional landing/takeoff detection

  cfg = ContactSensorCfg(..., track_air_time=True)
  sensor.compute_first_contact(dt)  # Landing events
  sensor.compute_first_air(dt)      # Takeoff events

Example
=======

  ContactSensorCfg(
    name="feet",
    primary=ContactMatch(mode="geom", pattern=["left_foot", "right_foot"], entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=2
  )
  # → 2 MuJoCo sensors (one per foot), each tracking 2 contacts → shape [B, 4, field_dim]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

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
  """Matching criteria for one side of a contact."""

  mode: Literal["geom", "body", "subtree"]
  """Matching mode."""
  pattern: str | list[str]
  """Regex string or list of regex strings. Expands when entity specified."""
  entity: str | None = None
  """Entity to search within for pattern expansion (None = treat pattern as literal)"""
  exclude: tuple[str, ...] = ()
  """Regex patterns to filter out from matches"""


@dataclass
class ContactSensorCfg(SensorCfg):
  """Contact sensor configuration.

  Primary pattern expands to N MuJoCo sensors. Each tracks num_slots contacts.
  Output shape: [B, N * num_slots, field_dim]

  fields: ("found", "force", "torque", "dist", "pos", "normal", "tangent")
    found=0 means no contact. Positive value = total matches before reduction.
    Normal points from primary→secondary. Empty slots are all zeros.

  reduce: How to pick top num_slots contacts
    "none" (fast, non-deterministic), "mindist", "maxforce", "netforce" (global frame)

  secondary_policy: When secondary pattern matches multiple ("first", "any", "error")
  track_air_time: Enable landing/takeoff detection
  global_frame: Rotate force/torque to global frame (needs normal+tangent in fields)
  """

  primary: ContactMatch
  secondary: ContactMatch | None = None
  fields: tuple[str, ...] = ("found", "force")
  reduce: Literal["none", "mindist", "maxforce", "netforce"] = "maxforce"
  num_slots: int = 1
  secondary_policy: Literal["first", "any", "error"] = "first"
  track_air_time: bool = False
  global_frame: bool = False
  debug: bool = False

  def build(self) -> ContactSensor:
    return ContactSensor(self)


@dataclass
class _ContactSlot:
  """Internal: maps one MuJoCo sensor to its sensordata buffer location."""

  name: str
  sensor_name: str
  data_slice: slice | None = None
  data_fields: tuple[str, ...] = ()
  field_offsets: dict[str, tuple[int, int]] | None = None
  total_dim: int = 0


@dataclass
class _AirTimeState:
  """Internal: air time tracking state."""

  current_air_time: torch.Tensor
  last_air_time: torch.Tensor
  current_contact_time: torch.Tensor
  last_contact_time: torch.Tensor
  last_time: torch.Tensor


@dataclass
class ContactData:
  """Contact sensor output. Shape: [B, N, ...].

  Fields (only requested fields are populated):
    found [B, N]: 0=no contact, positive=total match count
    force [B, N, 3]: Contact frame (global if reduce="netforce")
    torque [B, N, 3]: Contact frame (global if reduce="netforce")
    dist [B, N]: Penetration depth
    pos [B, N, 3]: Global frame
    normal [B, N, 3]: Global frame, primary→secondary
    tangent [B, N, 3]: Global frame (tangent2 = cross(normal, tangent))

  Air time (if track_air_time=True):
    current_air_time, last_air_time, current_contact_time, last_contact_time [B, N]
  """

  found: torch.Tensor | None = None
  force: torch.Tensor | None = None
  normal: torch.Tensor | None = None
  pos: torch.Tensor | None = None
  torque: torch.Tensor | None = None
  dist: torch.Tensor | None = None
  tangent: torch.Tensor | None = None

  current_air_time: torch.Tensor | None = None
  last_air_time: torch.Tensor | None = None
  current_contact_time: torch.Tensor | None = None
  last_contact_time: torch.Tensor | None = None


class ContactSensor(Sensor[ContactData]):
  """Contact sensor with pattern expansion."""

  def __init__(self, cfg: ContactSensorCfg) -> None:
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

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    """Expand patterns and add MuJoCo contact sensors to scene spec."""
    self._slots.clear()

    primary_names = self._resolve_primary_names(entities, self.cfg.primary)
    if self.cfg.secondary is None or self.cfg.secondary_policy == "any":
      secondary_name = None
    else:
      secondary_name = self._resolve_single_secondary(
        entities, self.cfg.secondary, self.cfg.secondary_policy
      )

    for slot_idx, prim in enumerate(primary_names):
      slot_name = f"{self.cfg.name}_slot{slot_idx}"

      self._add_contact_sensor_to_spec(scene_spec, slot_name, prim, secondary_name)

      field_offsets: dict[str, tuple[int, int]] = {}
      offset = 0
      for field in self.cfg.fields:
        dim = _CONTACT_DATA_DIMS[field]
        field_offsets[field] = (offset, offset + dim)
        offset += dim

      self._slots.append(
        _ContactSlot(
          name=prim,
          sensor_name=slot_name,
          data_fields=self.cfg.fields,
          field_offsets=field_offsets,
          total_dim=offset,
        )
      )

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    """Map sensors to sensordata buffer and allocate air time state."""
    del model  # Unused.

    if not self._slots:
      raise RuntimeError(
        f"There was an error initializing contact sensor '{self.cfg.name}'"
      )

    for slot in self._slots:
      sensor = mj_model.sensor(slot.sensor_name)
      start = sensor.adr[0]
      dim = sensor.dim[0]
      slot.data_slice = slice(start, start + dim)

    self._data = data
    self._device = device

    if self.cfg.track_air_time:
      n_envs = data.time.shape[0]
      n_slots = len(self._slots)
      self._air_time_state = _AirTimeState(
        current_air_time=torch.zeros((n_envs, n_slots), device=device),
        last_air_time=torch.zeros((n_envs, n_slots), device=device),
        current_contact_time=torch.zeros((n_envs, n_slots), device=device),
        last_contact_time=torch.zeros((n_envs, n_slots), device=device),
        last_time=torch.zeros((n_envs,), device=device),
      )

  @property
  def data(self) -> ContactData:
    """Current contact data. Shape: [B, N * num_slots, field_dim]."""
    out = self._extract_sensor_data()

    if self._air_time_state is not None:
      out.current_air_time = self._air_time_state.current_air_time
      out.last_air_time = self._air_time_state.last_air_time
      out.current_contact_time = self._air_time_state.current_contact_time
      out.last_contact_time = self._air_time_state.last_contact_time

    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset air time state for specified envs (or all if None)."""
    if self._air_time_state is None:
      return

    if env_ids is None:
      env_ids = slice(None)

    self._air_time_state.current_air_time[env_ids] = 0.0
    self._air_time_state.last_air_time[env_ids] = 0.0
    self._air_time_state.current_contact_time[env_ids] = 0.0
    self._air_time_state.last_contact_time[env_ids] = 0.0
    if self._data is not None:
      self._air_time_state.last_time[env_ids] = self._data.time[env_ids]

  def update(self, dt: float) -> None:
    """Update air time tracking (call once per step after physics forward)."""
    del dt  # Unused.
    if self._air_time_state is not None:
      self._update_air_time_tracking()

  def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
    """Landing events: contacts established within last dt seconds. [B, N] bool."""
    if self._air_time_state is None:
      raise RuntimeError(
        f"Sensor '{self.cfg.name}' must have track_air_time=True "
        "to use compute_first_contact"
      )
    is_in_contact = self._air_time_state.current_contact_time > 0.0
    within_dt = self._air_time_state.current_contact_time < (dt + abs_tol)
    return is_in_contact & within_dt

  def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
    """Takeoff events: contacts broken within last dt seconds. [B, N] bool."""
    if self._air_time_state is None:
      raise RuntimeError(
        f"Sensor '{self.cfg.name}' must have track_air_time=True "
        "to use compute_first_air"
      )
    is_in_air = self._air_time_state.current_air_time > 0.0
    within_dt = self._air_time_state.current_air_time < (dt + abs_tol)
    return is_in_air & within_dt

  def _extract_sensor_data(self) -> ContactData:
    if not self._slots or self._data is None:
      raise RuntimeError(f"Sensor '{self.cfg.name}' not initialized")

    field_names = self._slots[0].data_fields
    field_chunks: dict[str, list[torch.Tensor]] = {f: [] for f in field_names}
    for slot in self._slots:
      assert slot.data_slice is not None and slot.field_offsets is not None
      raw = self._data.sensordata[:, slot.data_slice]  # [B, slot_dim_total]
      per_slot_dim = slot.total_dim
      n_int = raw.size(1) // per_slot_dim
      arr = raw.view(raw.size(0), n_int, per_slot_dim)  # [B, n_int, slot_dim_total]
      for field in slot.data_fields:
        a, b = slot.field_offsets[field]
        field_chunks[field].append(arr[:, :, a:b])  # [B, n_int, d]

    out = ContactData()
    for field, chunks in field_chunks.items():
      cat = torch.cat(chunks, dim=1)  # [B, N, d]
      if cat.size(-1) == 1:  # squeeze scalar fields -> [B, N]
        cat = cat.squeeze(-1)
      setattr(out, field, cat)

    if self.cfg.global_frame and self.cfg.reduce != "netforce":
      out = self._transform_to_global_frame(out)

    return out

  def _transform_to_global_frame(self, data: ContactData) -> ContactData:
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

    is_contact = contact_data.found > 0

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

  def _resolve_primary_names(
    self, entities: dict[str, Entity], match: ContactMatch
  ) -> list[str]:
    if match.entity in (None, ""):
      return [match.pattern] if isinstance(match.pattern, str) else match.pattern

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
      compiled = [re.compile(p) for p in excludes]
      names = [n for n in names if not any(rx.search(n) for rx in compiled)]

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

    if isinstance(match.pattern, list):
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
  ) -> None:
    data_bits = sum(1 << _CONTACT_DATA_MAP[f] for f in self.cfg.fields)
    reduce_mode = _CONTACT_REDUCE_MAP[self.cfg.reduce]
    intprm = [data_bits, reduce_mode, self.cfg.num_slots]

    primary_entity = self.cfg.primary.entity
    if primary_entity and primary_entity != "":
      prefixed_primary = f"{primary_entity}/{primary_name}"
    else:
      prefixed_primary = primary_name

    kwargs = {
      "name": sensor_name,
      "type": mujoco.mjtSensor.mjSENS_CONTACT,
      "objtype": _MODE_TO_OBJTYPE[self.cfg.primary.mode],
      "objname": prefixed_primary,
      "intprm": intprm,
    }

    if secondary_name is not None:
      assert self.cfg.secondary is not None
      secondary_entity = self.cfg.secondary.entity
      if secondary_entity and secondary_entity != "":
        prefixed_secondary = f"{secondary_entity}/{secondary_name}"
      else:
        prefixed_secondary = secondary_name
      kwargs["reftype"] = _MODE_TO_OBJTYPE[self.cfg.secondary.mode]
      kwargs["refname"] = prefixed_secondary

    if self.cfg.debug:

      def _ename(v):
        return getattr(v, "name", str(v))

      objtype_name = _ename(kwargs["objtype"]).removeprefix("mjOBJ_")
      reftype_val = kwargs.get("reftype")
      refname_val = kwargs.get("refname")
      reftype_name = (
        _ename(reftype_val).removeprefix("mjOBJ_")
        if reftype_val is not None
        else "<any>"
      )

      ref_str = "<any>" if refname_val is None else f"{reftype_name}:{refname_val}"

      print(
        "Adding contact sensor\n"
        f"  name    : {sensor_name}\n"
        f"  object  : {objtype_name}:{kwargs['objname']}\n"
        f"  ref     : {ref_str}\n"
        f"  fields  : {self.cfg.fields}  bits=0b{intprm[0]:b}\n"
        f"  reduce  : {self.cfg.reduce}  num_slots={self.cfg.num_slots}"
      )

    scene_spec.add_sensor(**kwargs)
