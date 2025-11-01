"""Sensors that wrap MuJoCo builtin sensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg

SensorType = Literal[
  # Site sensors.
  "accelerometer",
  "velocimeter",
  "gyro",
  "force",
  "torque",
  "magnetometer",
  "rangefinder",
  # Joint sensors.
  "jointpos",
  "jointvel",
  "jointlimitpos",
  "jointlimitvel",
  "jointlimitfrc",
  "jointactuatorfrc",
  # Tendon sensors.
  "tendonpos",
  "tendonvel",
  "tendonactuatorfrc",
  # Actuator sensors.
  "actuatorpos",
  "actuatorvel",
  "actuatorfrc",
  # Frame sensors.
  "framepos",
  "framequat",
  "framexaxis",
  "frameyaxis",
  "framezaxis",
  "framelinvel",
  "frameangvel",
  "framelinacc",
  "frameangacc",
  # Subtree sensors.
  "subtreecom",
  "subtreelinvel",
  "subtreeangmom",
  # Misc.
  "e_potential",
  "e_kinetic",
  "clock",
]

_SENSOR_TYPE_MAP = {
  # Site sensors.
  "accelerometer": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
  "velocimeter": mujoco.mjtSensor.mjSENS_VELOCIMETER,
  "gyro": mujoco.mjtSensor.mjSENS_GYRO,
  "force": mujoco.mjtSensor.mjSENS_FORCE,
  "torque": mujoco.mjtSensor.mjSENS_TORQUE,
  "magnetometer": mujoco.mjtSensor.mjSENS_MAGNETOMETER,
  "rangefinder": mujoco.mjtSensor.mjSENS_RANGEFINDER,
  # Joint sensors.
  "jointpos": mujoco.mjtSensor.mjSENS_JOINTPOS,
  "jointvel": mujoco.mjtSensor.mjSENS_JOINTVEL,
  "jointlimitpos": mujoco.mjtSensor.mjSENS_JOINTLIMITPOS,
  "jointlimitvel": mujoco.mjtSensor.mjSENS_JOINTLIMITVEL,
  "jointlimitfrc": mujoco.mjtSensor.mjSENS_JOINTLIMITFRC,
  "jointactuatorfrc": mujoco.mjtSensor.mjSENS_JOINTACTFRC,
  # Tendon sensors.
  "tendonpos": mujoco.mjtSensor.mjSENS_TENDONPOS,
  "tendonvel": mujoco.mjtSensor.mjSENS_TENDONVEL,
  "tendonactuatorfrc": mujoco.mjtSensor.mjSENS_TENDONACTFRC,
  # Actuator sensors.
  "actuatorpos": mujoco.mjtSensor.mjSENS_ACTUATORPOS,
  "actuatorvel": mujoco.mjtSensor.mjSENS_ACTUATORVEL,
  "actuatorfrc": mujoco.mjtSensor.mjSENS_ACTUATORFRC,
  # Frame sensors.
  "framepos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
  "framequat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
  "framexaxis": mujoco.mjtSensor.mjSENS_FRAMEXAXIS,
  "frameyaxis": mujoco.mjtSensor.mjSENS_FRAMEYAXIS,
  "framezaxis": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
  "framelinvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
  "frameangvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
  "framelinacc": mujoco.mjtSensor.mjSENS_FRAMELINACC,
  "frameangacc": mujoco.mjtSensor.mjSENS_FRAMEANGACC,
  # Subtree sensors.
  "subtreecom": mujoco.mjtSensor.mjSENS_SUBTREECOM,
  "subtreelinvel": mujoco.mjtSensor.mjSENS_SUBTREELINVEL,
  "subtreeangmom": mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
  # Misc.
  "clock": mujoco.mjtSensor.mjSENS_CLOCK,
  "e_potential": mujoco.mjtSensor.mjSENS_E_POTENTIAL,
  "e_kinetic": mujoco.mjtSensor.mjSENS_E_KINETIC,
}

_OBJECT_TYPE_MAP = {
  "body": mujoco.mjtObj.mjOBJ_BODY,
  "xbody": mujoco.mjtObj.mjOBJ_XBODY,
  "joint": mujoco.mjtObj.mjOBJ_JOINT,
  "geom": mujoco.mjtObj.mjOBJ_GEOM,
  "site": mujoco.mjtObj.mjOBJ_SITE,
  "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
  "tendon": mujoco.mjtObj.mjOBJ_TENDON,
  "camera": mujoco.mjtObj.mjOBJ_CAMERA,
}

_SENSORS_REQUIRING_SITE = {
  "accelerometer",
  "velocimeter",
  "gyro",
  "force",
  "torque",
  "magnetometer",
  "rangefinder",
}

_SENSORS_REQUIRING_SPATIAL_FRAME = {
  "framepos",
  "framequat",
  "framexaxis",
  "frameyaxis",
  "framezaxis",
  "framelinvel",
  "frameangvel",
  "framelinacc",
  "frameangacc",
}

_SENSORS_REQUIRING_BODY = {
  "subtreecom",
  "subtreelinvel",
  "subtreeangmom",
}

_SENSOR_OBJECT_REQUIREMENTS = {
  "jointpos": "joint",
  "jointvel": "joint",
  "jointlimitpos": "joint",
  "jointlimitvel": "joint",
  "jointlimitfrc": "joint",
  "jointactuatorfrc": "joint",
  "tendonpos": "tendon",
  "tendonvel": "tendon",
  "tendonactuatorfrc": "tendon",
  "actuatorpos": "actuator",
  "actuatorvel": "actuator",
  "actuatorfrc": "actuator",
}

_SPATIAL_FRAME_TYPES = {"body", "xbody", "geom", "site", "camera"}
_SENSORS_ALLOWING_REF = {
  "framepos",
  "framequat",
  "framexaxis",
  "frameyaxis",
  "framezaxis",
  "framelinvel",
  "frameangvel",
  "framelinacc",
  "frameangacc",
}


@dataclass
class ObjRef:
  """Reference to a MuJoCo object (body, joint, site, etc.).

  Used to specify which object a sensor is attached to and its frame of reference.
  The entity field allows scoping objects to specific entity namespaces.
  """

  type: Literal[
    "body", "xbody", "joint", "geom", "site", "actuator", "tendon", "camera"
  ]
  """Type of the object."""
  name: str
  """Name of the object."""
  entity: str | None = None
  """Optional entity prefix for the object name."""

  def prefixed_name(self) -> str:
    """Get the full name with entity prefix if applicable."""
    if self.entity:
      return f"{self.entity}/{self.name}"
    return self.name


@dataclass
class BuiltinSensorCfg(SensorCfg):
  sensor_type: SensorType
  """Which builtin sensor to use."""
  obj: ObjRef | None = None
  """The type and name of the object the sensor is attached to."""
  ref: ObjRef | None = None
  """The type and name of object to which the frame-of-reference is attached to."""
  cutoff: float = 0.0
  """When this value is positive, it limits the absolute value of the sensor output."""

  def __post_init__(self) -> None:
    # Auto-prefix sensor name if it references an entity.
    if self.obj is not None and self.obj.entity is not None:
      self.name = f"{self.obj.entity}/{self.name}"

    if self.sensor_type in _SENSORS_REQUIRING_SITE:
      if self.obj is None:
        raise ValueError(
          f"Sensor type '{self.sensor_type}' requires obj with type='site'"
        )
      if self.obj.type != "site":
        raise ValueError(
          f"Sensor type '{self.sensor_type}' requires obj.type='site', got "
          f"'{self.obj.type}'"
        )

    elif self.sensor_type in _SENSORS_REQUIRING_SPATIAL_FRAME:
      if self.obj is None:
        raise ValueError(
          f"Sensor type '{self.sensor_type}' requires obj with spatial frame"
        )
      if self.obj.type not in _SPATIAL_FRAME_TYPES:
        raise ValueError(
          f"Sensor type '{self.sensor_type}' requires obj.type in "
          f"{_SPATIAL_FRAME_TYPES}, got '{self.obj.type}'"
        )

    elif self.sensor_type in _SENSORS_REQUIRING_BODY:
      if self.obj is None:
        raise ValueError(
          f"Sensor type '{self.sensor_type}' requires obj with type='body'"
        )
      if self.obj.type != "body":
        raise ValueError(
          f"Sensor type '{self.sensor_type}' requires obj.type='body', "
          f"got '{self.obj.type}'"
        )

    elif self.sensor_type in _SENSOR_OBJECT_REQUIREMENTS:
      required_type = _SENSOR_OBJECT_REQUIREMENTS[self.sensor_type]
      if self.obj is None:
        raise ValueError(
          f"Sensor type '{self.sensor_type}' requires obj with type='{required_type}'"
        )
      if self.obj.type != required_type:
        raise ValueError(
          f"Sensor type '{self.sensor_type}' requires obj.type='{required_type}', "
          f"got '{self.obj.type}'"
        )

    if self.ref is not None and self.sensor_type not in _SENSORS_ALLOWING_REF:
      raise ValueError(
        f"Sensor type '{self.sensor_type}' does not support ref specification"
      )

  def build(self) -> BuiltinSensor:
    return BuiltinSensor(self)


class BuiltinSensor(Sensor[torch.Tensor]):
  """Wrapper over MuJoCo builtin sensors.

  Can add a new sensor to the spec, or wrap an existing sensor from entity XML.
  Returns raw MuJoCo sensordata as torch.Tensor with shape depending on sensor type
  (e.g., accelerometer: (N, 3), framequat: (N, 4)).
  """

  def __init__(
    self, cfg: BuiltinSensorCfg | None = None, name: str | None = None
  ) -> None:
    if cfg is not None:
      self._name = cfg.name
      self.cfg: BuiltinSensorCfg | None = cfg
    else:
      assert name is not None, "Must provide either cfg or name"
      self._name = name
      self.cfg = None
    self._data: mjwarp.Data | None = None
    self._data_view: torch.Tensor | None = None

  @classmethod
  def from_existing(cls, name: str) -> BuiltinSensor:
    """Wrap an existing sensor already defined in entity XML."""
    return cls(cfg=None, name=name)

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del entities
    if self.cfg is None:
      return

    # Check for duplicate sensors.
    for sensor in scene_spec.sensors:
      if sensor.name == self.cfg.name:
        is_entity_scoped = self.cfg.obj is not None and self.cfg.obj.entity is not None
        if is_entity_scoped:
          raise ValueError(
            f"Sensor '{self.cfg.name}' is defined in both entity XML and scene config. "
            f"Remove the sensor definition from the entity XML file, or remove the "
            f"BuiltinSensorCfg from scene.sensors."
          )
        else:
          raise ValueError(
            f"Sensor '{self.cfg.name}' already exists in the scene. "
            f"Rename this sensor to avoid conflicts."
          )

    # Add sensor to spec.
    kwargs = {
      "name": self.cfg.name,
      "type": _SENSOR_TYPE_MAP[self.cfg.sensor_type],
    }
    if self.cfg.obj is not None:
      kwargs["objtype"] = _OBJECT_TYPE_MAP[self.cfg.obj.type]
      kwargs["objname"] = self.cfg.obj.prefixed_name()
    if self.cfg.ref is not None:
      kwargs["reftype"] = _OBJECT_TYPE_MAP[self.cfg.ref.type]
      kwargs["refname"] = self.cfg.ref.prefixed_name()
    if self.cfg.cutoff > 0:
      kwargs["cutoff"] = self.cfg.cutoff
    scene_spec.add_sensor(**kwargs)

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    del model, device
    self._data = data
    sensor = mj_model.sensor(self._name)
    start = sensor.adr[0]
    dim = sensor.dim[0]
    self._data_view = self._data.sensordata[:, start : start + dim]

  @property
  def data(self) -> torch.Tensor:
    assert self._data_view is not None
    return self._data_view
