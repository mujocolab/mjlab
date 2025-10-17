from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.sensor.base import Sensor, SensorCfg
from mjlab.sensor.builtin import BuiltinContactSensor, BuiltinContactSensorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass
class ContactSensorCfg(SensorCfg):
  """Configuration for a regex-based contact sensor."""

  name: str
  entity_name: str
  geom1: str | None = None
  geom2: str | None = None
  body1: str | None = None
  body2: str | None = None
  subtree1: str | None = None
  subtree2: str | None = None
  secondary_entity: str | None = None
  data: tuple[
    Literal["found", "force", "torque", "dist", "pos", "normal", "tangent"], ...
  ] = ("found",)
  reduce: Literal["none", "mindist", "maxforce", "netforce"] = "maxforce"
  track_air_time: bool = False
  force_threshold: float = 1e-3

  def build(self) -> Sensor:
    return ContactSensor(self)


@dataclass
class _ContactSlot:
  name: str
  sensor: BuiltinContactSensor


@dataclass
class _AirTimeState:
  current_air_time: torch.Tensor
  last_air_time: torch.Tensor
  current_contact_time: torch.Tensor
  last_contact_time: torch.Tensor


class ContactSensorData:
  """Structured view over the per-slot builtin sensors."""

  def __init__(
    self,
    slots: list[_ContactSlot],
    air_time_state: _AirTimeState | None = None,
  ) -> None:
    self._slots = slots
    self._air_time_state = air_time_state

  def _concat(self, field: str) -> torch.Tensor:
    if not self._slots:
      raise RuntimeError("ContactSensor has no resolved slots.")
    tensors = [getattr(slot.sensor.data, field) for slot in self._slots]
    return torch.cat(tensors, dim=1)

  @property
  def found(self) -> torch.Tensor:
    """Shape: (num_envs, num_slots, 1)."""
    return self._concat("found")

  @property
  def force(self) -> torch.Tensor:
    """Shape: (num_envs, num_slots, 3)."""
    return self._concat("force")

  @property
  def torque(self) -> torch.Tensor:
    """Shape: (num_envs, num_slots, 3)."""
    return self._concat("torque")

  @property
  def dist(self) -> torch.Tensor:
    """Shape: (num_envs, num_slots, 1)."""
    return self._concat("dist")

  @property
  def pos(self) -> torch.Tensor:
    """Shape: (num_envs, num_slots, 3)."""
    return self._concat("pos")

  @property
  def normal(self) -> torch.Tensor:
    """Shape: (num_envs, num_slots, 3)."""
    return self._concat("normal")

  @property
  def count(self) -> torch.Tensor:
    """Shape: (num_envs, num_slots)."""
    if not self._slots:
      raise RuntimeError("ContactSensor has no resolved slots.")
    counts = [slot.sensor.count for slot in self._slots]
    return torch.stack(counts, dim=1)

  @property
  def current_air_time(self) -> torch.Tensor | None:
    return (
      None if self._air_time_state is None else self._air_time_state.current_air_time
    )

  @property
  def last_air_time(self) -> torch.Tensor | None:
    return None if self._air_time_state is None else self._air_time_state.last_air_time

  @property
  def current_contact_time(self) -> torch.Tensor | None:
    if self._air_time_state is None:
      return None
    return self._air_time_state.current_contact_time

  @property
  def last_contact_time(self) -> torch.Tensor | None:
    if self._air_time_state is None:
      return None
    return self._air_time_state.last_contact_time


class ContactSensor(Sensor):
  """Flexible contact sensor with pattern matching and per-slot data."""

  def __init__(self, cfg: ContactSensorCfg) -> None:
    super().__init__(cfg)
    self.name = cfg.name
    self._cfg = cfg
    self._slots: list[_ContactSlot] = []
    self._slot_names: list[str] = []
    self._slot_map: dict[str, int] = {}
    self._data_view: ContactSensorData | None = None
    self._air_time_state: _AirTimeState | None = None

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    self._slots.clear()
    self._slot_names.clear()
    self._slot_map.clear()

    entity1, entity2, entity2_name = self._resolve_entities(entities)

    matches_side1 = sorted(
      set(
        self._resolve_side(
          entity1,
          self._cfg.geom1,
          self._cfg.body1,
          self._cfg.subtree1,
          "side 1",
        )
      )
    )
    if not matches_side1:
      raise ValueError(
        f"ContactSensor '{self.name}' resolved zero slots for pattern "
        f"'{self._cfg.geom1 or self._cfg.body1 or self._cfg.subtree1}'."
      )

    if entity2 is None:
      matches_side2 = None
      side2_type = self._get_side_type(
        self._cfg.geom2, self._cfg.body2, self._cfg.subtree2
      )
      side2_name = self._cfg.geom2 or self._cfg.body2 or self._cfg.subtree2
    else:
      matches_side2 = sorted(
        set(
          self._resolve_side(
            entity2,
            self._cfg.geom2,
            self._cfg.body2,
            self._cfg.subtree2,
            "side 2",
          )
        )
      )
      side2_type = None
      side2_name = None

    side1_type = self._get_side_type(
      self._cfg.geom1, self._cfg.body1, self._cfg.subtree1
    )
    data_fields = self._prepare_data_fields()

    for slot_index, match1 in enumerate(matches_side1):
      builtin_cfg = self._build_builtin_cfg(
        slot_index=slot_index,
        match1=match1,
        side1_type=side1_type,
        matches_side2=matches_side2,
        side2_type=side2_type,
        side2_name=side2_name,
        entity2_name=entity2_name,
        data_fields=data_fields,
      )
      builtin_sensor = BuiltinContactSensor(builtin_cfg)
      builtin_sensor.edit_spec(scene_spec, entities)
      self._slots.append(_ContactSlot(name=match1, sensor=builtin_sensor))
      self._slot_names.append(match1)

    self._slot_map = {name: idx for idx, name in enumerate(self._slot_names)}

  def _resolve_entities(
    self, entities: dict[str, Entity]
  ) -> tuple[Entity, Entity | None, str]:
    if self._cfg.entity_name not in entities:
      raise ValueError(
        f"Entity '{self._cfg.entity_name}' not found in scene. "
        f"Available: {list(entities.keys())}"
      )
    entity1 = entities[self._cfg.entity_name]

    entity2_name = (
      self._cfg.secondary_entity
      if self._cfg.secondary_entity is not None
      else self._cfg.entity_name
    )

    if entity2_name == "":
      return entity1, None, entity2_name

    if entity2_name not in entities:
      raise ValueError(
        f"Secondary entity '{entity2_name}' not found in scene. "
        f"Available: {list(entities.keys())}"
      )
    return entity1, entities[entity2_name], entity2_name

  def _prepare_data_fields(self) -> tuple[str, ...]:
    fields = list(self._cfg.data)
    for required in ("found",):
      if required not in fields:
        fields.insert(0, required)
    if self._cfg.track_air_time and "force" not in fields:
      fields.append("force")

    deduped: list[str] = []
    for field in fields:
      if field not in deduped:
        deduped.append(field)
    return tuple(deduped)

  def _build_builtin_cfg(
    self,
    *,
    slot_index: int,
    match1: str,
    side1_type: str,
    matches_side2: list[str] | None,
    side2_type: str | None,
    side2_name: str | None,
    entity2_name: str,
    data_fields: tuple[str, ...],
  ) -> BuiltinContactSensorCfg:
    params: dict[str, object] = {
      "name": f"{self.name}_slot{slot_index}",
      "entity_name": self._cfg.entity_name,
      "num": 1,
      "data": data_fields,
      "reduce": self._cfg.reduce,
    }

    if side1_type == "geom":
      params["geom1"] = match1
    elif side1_type == "body":
      params["body1"] = match1
    elif side1_type == "subtree":
      params["subtree1"] = match1

    if matches_side2 is None:
      params["secondary_entity"] = ""
      if side2_type == "geom" and side2_name:
        params["geom2"] = side2_name
      elif side2_type == "body" and side2_name:
        params["body2"] = side2_name
      elif side2_type == "subtree" and side2_name:
        params["subtree2"] = side2_name
    elif len(matches_side2) == 1:
      params["secondary_entity"] = entity2_name
      side2_type_resolved = self._get_side_type(
        self._cfg.geom2, self._cfg.body2, self._cfg.subtree2
      )
      target = matches_side2[0]
      if side2_type_resolved == "geom":
        params["geom2"] = target
      elif side2_type_resolved == "body":
        params["body2"] = target
      elif side2_type_resolved == "subtree":
        params["subtree2"] = target
    else:
      raise ValueError(
        "Side 2 pattern matched multiple targets. ContactSensor requires a single "
        "match. Narrow the pattern or create separate sensors per match."
      )

    return BuiltinContactSensorCfg(**params)  # type: ignore[arg-type]

  def _resolve_side(
    self,
    entity: Entity,
    geom_pattern: str | None,
    body_pattern: str | None,
    subtree: str | None,
    side_label: str,
  ) -> list[str]:
    patterns_specified = sum(
      x is not None for x in [geom_pattern, body_pattern, subtree]
    )
    if patterns_specified == 0:
      raise ValueError(
        f"{side_label}: Must specify exactly one of geom, body, or subtree pattern"
      )
    if patterns_specified > 1:
      raise ValueError(
        f"{side_label}: Must specify exactly one of geom, body, or subtree pattern "
        f"(got {patterns_specified})"
      )

    entity_name = getattr(entity.cfg, "name", str(entity.cfg))

    if geom_pattern is not None:
      _, names = entity.find_geoms(geom_pattern)
      if not names:
        raise ValueError(
          f"{side_label}: Geom pattern '{geom_pattern}' matched no geoms in entity '{entity_name}'"
        )
      return names
    if body_pattern is not None:
      _, names = entity.find_bodies(body_pattern)
      if not names:
        raise ValueError(
          f"{side_label}: Body pattern '{body_pattern}' matched no bodies in entity '{entity_name}'"
        )
      return names
    if subtree is not None:
      return [subtree]
    raise ValueError(f"{side_label}: Must specify geom, body, or subtree pattern")

  def _get_side_type(
    self,
    geom_pattern: str | None,
    body_pattern: str | None,
    subtree: str | None,
  ) -> str:
    if geom_pattern is not None:
      return "geom"
    if body_pattern is not None:
      return "body"
    if subtree is not None:
      return "subtree"
    raise ValueError("Must specify geom, body, or subtree pattern")

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    if not self._slots:
      raise RuntimeError(
        f"ContactSensor '{self.name}' has no resolved slots. Did edit_spec run?"
      )

    for slot in self._slots:
      slot.sensor.initialize(mj_model, model, data, device)

    air_state: _AirTimeState | None = None
    if self._cfg.track_air_time:
      num_slots = len(self._slots)
      zeros = torch.zeros(data.nworld, num_slots, device=device, dtype=torch.float32)
      air_state = _AirTimeState(
        current_air_time=zeros.clone(),
        last_air_time=zeros.clone(),
        current_contact_time=zeros.clone(),
        last_contact_time=zeros.clone(),
      )

    self._air_time_state = air_state
    self._data_view = ContactSensorData(self._slots, air_state)

  @property
  def data(self) -> ContactSensorData:
    if self._data_view is None:
      raise RuntimeError(f"Sensor '{self.name}' not initialized")
    return self._data_view

  @property
  def slot_names(self) -> list[str]:
    return self._slot_names

  @property
  def slot_map(self) -> dict[str, int]:
    return self._slot_map

  @property
  def count(self) -> torch.Tensor:
    return self.data.count

  @property
  def current_air_time(self) -> torch.Tensor:
    if not self._cfg.track_air_time or self._air_time_state is None:
      raise AttributeError(
        f"Sensor '{self.name}' does not have air time tracking enabled. "
        "Set track_air_time=True in the config."
      )
    return self._air_time_state.current_air_time

  @property
  def last_air_time(self) -> torch.Tensor:
    if not self._cfg.track_air_time or self._air_time_state is None:
      raise AttributeError(
        f"Sensor '{self.name}' does not have air time tracking enabled. "
        "Set track_air_time=True in the config."
      )
    return self._air_time_state.last_air_time

  def update(self, dt: float) -> None:
    if not self._cfg.track_air_time or self._air_time_state is None:
      return

    force = self.data.force
    is_contact = force.norm(dim=-1) > self._cfg.force_threshold

    current_air_time = self._air_time_state.current_air_time
    last_air_time = self._air_time_state.last_air_time
    current_contact_time = self._air_time_state.current_contact_time
    last_contact_time = self._air_time_state.last_contact_time

    was_in_air = current_air_time > 0
    was_in_contact = current_contact_time > 0
    is_first_contact = was_in_air & is_contact
    is_first_detached = was_in_contact & ~is_contact

    self._air_time_state.last_air_time = torch.where(
      is_first_contact,
      current_air_time + dt,
      last_air_time,
    )

    self._air_time_state.current_air_time = torch.where(
      ~is_contact, current_air_time + dt, torch.zeros_like(current_air_time)
    )

    self._air_time_state.last_contact_time = torch.where(
      is_first_detached,
      current_contact_time + dt,
      last_contact_time,
    )

    self._air_time_state.current_contact_time = torch.where(
      is_contact,
      current_contact_time + dt,
      torch.zeros_like(current_contact_time),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if not self._cfg.track_air_time or self._air_time_state is None:
      return

    if env_ids is None:
      env_ids = slice(None)

    self._air_time_state.last_air_time[env_ids] = 0.0
    self._air_time_state.current_air_time[env_ids] = 0.0
    self._air_time_state.last_contact_time[env_ids] = 0.0
    self._air_time_state.current_contact_time[env_ids] = 0.0
