"""Builtin MuJoCo sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.sensor.base import Sensor, SensorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


def _prefix(name: str | None, entity: str | None) -> str | None:
  """Prefix name with entity when provided."""
  if not name or not entity:
    return name
  return f"{entity}/{name}"


# ============================================================================
# Simple Builtin Sensors (gyro, accelerometer, etc.)
# ============================================================================

_SENSOR_TYPE_MAP = {
  "accelerometer": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
  "frameangvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
  "framelinvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
  "framepos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
  "framequat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
  "framezaxis": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
  "gyro": mujoco.mjtSensor.mjSENS_GYRO,
  "subtreeangmom": mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
  "upvector": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
  "velocimeter": mujoco.mjtSensor.mjSENS_VELOCIMETER,
}

_OBJECT_TYPE_MAP = {
  "body": mujoco.mjtObj.mjOBJ_BODY,
  "geom": mujoco.mjtObj.mjOBJ_GEOM,
  "site": mujoco.mjtObj.mjOBJ_SITE,
  "xbody": mujoco.mjtObj.mjOBJ_XBODY,
}


@dataclass
class BuiltinSensorCfg(SensorCfg):
  """Configuration for builtin MuJoCo sensors."""

  name: str
  entity_name: str
  sensor_type: Literal[
    "accelerometer",
    "frameangvel",
    "framelinvel",
    "framepos",
    "framequat",
    "framezaxis",
    "gyro",
    "subtreeangmom",
    "upvector",
    "velocimeter",
  ]
  objtype: Literal["body", "geom", "site", "xbody"]
  objname: str
  reftype: Literal["body", "geom", "site", "xbody"] | None = None
  refname: str | None = None
  obj_entity: str | None = None
  ref_entity: str | None = None

  def build(self) -> Sensor:
    """Build sensor instance from this config."""
    return BuiltinSensor(self)

  def edit_spec(self, scene_spec: mujoco.MjSpec) -> None:
    if (self.reftype is None) != (self.refname is None):
      raise ValueError("reftype and refname must both be set or both be None")

    obj_entity = self.obj_entity or self.entity_name
    ref_entity = self.ref_entity if self.ref_entity is not None else self.entity_name

    kwargs = {
      "name": self.name,
      "type": _SENSOR_TYPE_MAP[self.sensor_type],
      "objtype": _OBJECT_TYPE_MAP[self.objtype],
      "objname": _prefix(self.objname, obj_entity),
    }

    if self.reftype is not None:
      kwargs["reftype"] = _OBJECT_TYPE_MAP[self.reftype]
      kwargs["refname"] = _prefix(self.refname, ref_entity)

    scene_spec.add_sensor(**kwargs)


class BuiltinSensor(Sensor):
  """Builtin MuJoCo sensor wrapper.

  Provides access to MuJoCo's native sensors like gyro, accelerometer, framelinvel, etc.
  Returns raw sensordata tensor directly from mujoco_warp.Data.sensordata.

  Use this for simple, stateless sensors that don't need additional computation.

  Example:
      >>> cfg = BuiltinSensorCfg(
      ...     name="base_gyro",
      ...     entity_name="robot",
      ...     sensor_type="gyro",
      ...     objtype="body",
      ...     objname="pelvis",
      ... )
      >>> sensor = cfg.build()
      >>> # After initialization:
      >>> gyro_reading = sensor.data  # Shape: (num_envs, 3)
  """

  def __init__(self, cfg: BuiltinSensorCfg) -> None:
    super().__init__(cfg)
    self.name = cfg.name
    self._cfg = cfg
    self._warp_data: mjwarp.Data | None = None
    self._start: int | None = None
    self._end: int | None = None

  def edit_spec(
    self,
    scene_spec: mujoco.MjSpec,
    entities: dict[str, Entity],
  ) -> None:
    del entities  # Unused.
    self._cfg.edit_spec(scene_spec)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    del model, device  # Unused
    sensor = mj_model.sensor(self.name)
    self._warp_data = data
    self._start = sensor.adr[0]
    self._end = self._start + sensor.dim[0]

  @property
  def data(self) -> torch.Tensor:
    """Raw sensor data tensor."""
    assert self._warp_data is not None, f"Sensor '{self.name}' not initialized"
    assert self._start is not None
    assert self._end is not None
    return self._warp_data.sensordata[:, self._start : self._end]


# ============================================================================
# Builtin Contact Sensor (raw MuJoCo contact sensor)
# ============================================================================

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

_CONTACT_DATA_DIMS = {
  "found": 1,
  "force": 3,
  "torque": 3,
  "dist": 1,
  "pos": 3,
  "normal": 3,
  "tangent": 3,
}


@dataclass
class BuiltinContactSensorCfg(SensorCfg):
  """Configuration for builtin MuJoCo contact sensor.

  See MuJoCo documentation for details:
  https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor-contact
  """

  name: str
  entity_name: str
  geom1: str | None = None
  geom2: str | None = None
  body1: str | None = None
  body2: str | None = None
  subtree1: str | None = None
  subtree2: str | None = None
  secondary_entity: str | None = None
  num: int = 1
  data: tuple[
    Literal["found", "force", "torque", "dist", "pos", "normal", "tangent"], ...
  ] = ("force",)
  reduce: Literal["none", "mindist", "maxforce", "netforce"] = "maxforce"

  def build(self) -> Sensor:
    """Build sensor instance from this config."""
    return BuiltinContactSensor(self)

  def edit_spec(self, scene_spec: mujoco.MjSpec) -> None:
    # Determine primary object.
    primary_count = sum(x is not None for x in [self.geom1, self.body1, self.subtree1])
    if primary_count != 1:
      raise ValueError("Must specify exactly one of: geom1, body1, subtree1")

    if self.geom1 is not None:
      objtype, objname = mujoco.mjtObj.mjOBJ_GEOM, self.geom1
    elif self.body1 is not None:
      objtype, objname = mujoco.mjtObj.mjOBJ_BODY, self.body1
    else:
      objtype, objname = mujoco.mjtObj.mjOBJ_XBODY, self.subtree1

    # Determine secondary object.
    secondary_count = sum(
      x is not None for x in [self.geom2, self.body2, self.subtree2]
    )
    if secondary_count > 1:
      raise ValueError("Must specify at most one of: geom2, body2, subtree2")

    if self.geom2 is not None:
      reftype, refname = mujoco.mjtObj.mjOBJ_GEOM, self.geom2
    elif self.body2 is not None:
      reftype, refname = mujoco.mjtObj.mjOBJ_BODY, self.body2
    elif self.subtree2 is not None:
      reftype, refname = mujoco.mjtObj.mjOBJ_XBODY, self.subtree2
    else:
      reftype, refname = None, None

    # Prefix with entities.
    primary_entity = self.entity_name
    secondary_entity = (
      self.secondary_entity if self.secondary_entity is not None else self.entity_name
    )

    # Build intprm: [data_bits, reduce_mode, num_contacts].
    data_bits = sum(1 << _CONTACT_DATA_MAP[d] for d in self.data)
    reduce_mode = _CONTACT_REDUCE_MAP[self.reduce]

    kwargs = {
      "name": self.name,
      "type": mujoco.mjtSensor.mjSENS_CONTACT,
      "objtype": objtype,
      "objname": _prefix(objname, primary_entity),
      "intprm": [data_bits, reduce_mode, self.num],
    }

    if reftype is not None and refname is not None:
      kwargs["reftype"] = reftype
      kwargs["refname"] = _prefix(refname, secondary_entity)

    scene_spec.add_sensor(**kwargs)


@dataclass
class BuiltinContactSensorData:
  """Structured data view for builtin contact sensor.

  Provides semantic access to contact sensor fields like force, position, normal, etc.
  See MuJoCo documentation for contact sensor details:
  https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor-contact
  """

  _data: mjwarp.Data = field(repr=False)
  start: int
  end: int  # Precomputed start + size
  data_fields: tuple[str, ...]
  num_contacts: int
  _total_dim: int = field(init=False)  # Cached total dimension per contact
  _field_offsets: dict[str, tuple[int, int]] = field(
    init=False
  )  # Cached (offset, dim) per field

  def __post_init__(self) -> None:
    """Precompute field offsets and total dimension."""
    self._total_dim = sum(_CONTACT_DATA_DIMS[f] for f in self.data_fields)
    self._field_offsets = {}
    offset = 0
    for field_name in self.data_fields:
      dim = _CONTACT_DATA_DIMS[field_name]
      self._field_offsets[field_name] = (offset, dim)
      offset += dim

  def _get_field_slice(self, field_name: str) -> torch.Tensor:
    """Get data for a specific field."""
    if field_name not in self._field_offsets:
      raise ValueError(
        f"Field '{field_name}' not in sensor data fields: {self.data_fields}"
      )
    offset, dim = self._field_offsets[field_name]
    raw = self._data.sensordata[:, self.start : self.end]
    raw = raw.view(-1, self.num_contacts, self._total_dim)
    return raw[:, :, offset : offset + dim]

  @property
  def found(self) -> torch.Tensor:
    """Contact found indicator. Shape: (num_envs, num_contacts, 1)

    Raises:
      ValueError: If sensor not configured with 'found' field.
    """
    return self._get_field_slice("found")

  @property
  def force(self) -> torch.Tensor:
    """Contact force. Shape: (num_envs, num_contacts, 3)

    Raises:
      ValueError: If sensor not configured with 'force' field.
    """
    return self._get_field_slice("force")

  @property
  def torque(self) -> torch.Tensor:
    """Contact torque. Shape: (num_envs, num_contacts, 3)

    Raises:
      ValueError: If sensor not configured with 'torque' field.
    """
    return self._get_field_slice("torque")

  @property
  def dist(self) -> torch.Tensor:
    """Contact distance. Shape: (num_envs, num_contacts, 1)

    Raises:
      ValueError: If sensor not configured with 'dist' field.
    """
    return self._get_field_slice("dist")

  @property
  def pos(self) -> torch.Tensor:
    """Contact position. Shape: (num_envs, num_contacts, 3)

    Raises:
      ValueError: If sensor not configured with 'pos' field.
    """
    return self._get_field_slice("pos")

  @property
  def normal(self) -> torch.Tensor:
    """Contact normal. Shape: (num_envs, num_contacts, 3)

    Raises:
      ValueError: If sensor not configured with 'normal' field.
    """
    return self._get_field_slice("normal")

  @property
  def count(self) -> torch.Tensor:
    """Total number of contacts per environment. Shape: (num_envs,)

    The 'found' field contains the total number of matching contacts.
    All filled slots have the same value, so we take the first slot.

    Raises:
      ValueError: If sensor not configured with 'found' field.
    """
    return self.found[:, 0, 0]


class BuiltinContactSensor(Sensor):
  """Builtin MuJoCo contact sensor.

  Wraps MuJoCo's native contact sensor and provides structured data access.

  Example:
      >>> cfg = BuiltinContactSensorCfg(
      ...     name="foot_contact",
      ...     entity_name="robot",
      ...     body1="foot",
      ...     body2="terrain",
      ...     secondary_entity="",
      ...     data=("found", "force", "normal"),
      ...     num=4,
      ...     reduce="maxforce",
      ... )
      >>> sensor = cfg.build()
      >>> # After initialization:
      >>> contact_force = sensor.data.force  # Shape: (num_envs, 4, 3)
      >>> contact_count = sensor.data.count  # Shape: (num_envs,)
  """

  def __init__(self, cfg: BuiltinContactSensorCfg) -> None:
    super().__init__(cfg)
    self.name = cfg.name
    self._data_view: BuiltinContactSensorData | None = None
    self._cfg = cfg

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del entities  # Unused.
    self._cfg.edit_spec(scene_spec)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    del model, device  # Unused
    sensor = mj_model.sensor(self.name)
    start = sensor.adr[0]
    self._data_view = BuiltinContactSensorData(
      _data=data,
      start=start,
      end=start + sensor.dim[0],
      data_fields=self._cfg.data,
      num_contacts=self._cfg.num,
    )

  @property
  def data(self) -> BuiltinContactSensorData:
    """Parsed contact sensor data."""
    assert self._data_view is not None, f"Sensor '{self.name}' not initialized"
    return self._data_view

  @property
  def count(self) -> torch.Tensor:
    """Shortcut for data.count. Shape: (num_envs,)"""
    return self.data.count

  @property
  def force(self) -> torch.Tensor:
    """Shortcut for data.force. Shape: (num_envs, num_contacts, 3)"""
    return self.data.force
