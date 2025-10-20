from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg

SensorType = Literal[
  "accelerometer",
  "velocimeter",
  "gyro",
  "force",
  "torque",
  "magnetometer",
  "rangefinder",
  "jointpos",
  "jointvel",
  "tendonpos",
  "tendonvel",
  "actuatorpos",
  "actuatorvel",
  "actuatorfrc",
  "jointlimitpos",
  "jointlimitvel",
  "jointlimitfrc",
  "framepos",
  "framequat",
  "framexaxis",
  "frameyaxis",
  "framezaxis",
  "framelinvel",
  "frameangvel",
  "framelinacc",
  "frameangacc",
  "subtreecom",
  "subtreelinvel",
  "subtreeangmom",
  "e_potential",
  "e_kinetic",
  "clock",
]
ObjRefType = Literal[
  "body", "xbody", "joint", "geom", "site", "actuator", "tendon", "camera"
]

_SENSOR_TYPE_MAP = {
  "accelerometer": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
  "velocimeter": mujoco.mjtSensor.mjSENS_VELOCIMETER,
  "gyro": mujoco.mjtSensor.mjSENS_GYRO,
  "force": mujoco.mjtSensor.mjSENS_FORCE,
  "torque": mujoco.mjtSensor.mjSENS_TORQUE,
  "magnetometer": mujoco.mjtSensor.mjSENS_MAGNETOMETER,
  "rangefinder": mujoco.mjtSensor.mjSENS_RANGEFINDER,
  "jointpos": mujoco.mjtSensor.mjSENS_JOINTPOS,
  "jointvel": mujoco.mjtSensor.mjSENS_JOINTVEL,
  "tendonpos": mujoco.mjtSensor.mjSENS_TENDONPOS,
  "tendonvel": mujoco.mjtSensor.mjSENS_TENDONVEL,
  "actuatorpos": mujoco.mjtSensor.mjSENS_ACTUATORPOS,
  "actuatorvel": mujoco.mjtSensor.mjSENS_ACTUATORVEL,
  "actuatorfrc": mujoco.mjtSensor.mjSENS_ACTUATORFRC,
  "jointlimitpos": mujoco.mjtSensor.mjSENS_JOINTLIMITPOS,
  "jointlimitvel": mujoco.mjtSensor.mjSENS_JOINTLIMITVEL,
  "jointlimitfrc": mujoco.mjtSensor.mjSENS_JOINTLIMITFRC,
  "framepos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
  "framequat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
  "framexaxis": mujoco.mjtSensor.mjSENS_FRAMEXAXIS,
  "frameyaxis": mujoco.mjtSensor.mjSENS_FRAMEYAXIS,
  "framezaxis": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
  "framelinvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
  "frameangvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
  "framelinacc": mujoco.mjtSensor.mjSENS_FRAMELINACC,
  "frameangacc": mujoco.mjtSensor.mjSENS_FRAMEANGACC,
  "subtreecom": mujoco.mjtSensor.mjSENS_SUBTREECOM,
  "subtreelinvel": mujoco.mjtSensor.mjSENS_SUBTREELINVEL,
  "subtreeangmom": mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
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


def _prefix_name(name: str, entity: str | None) -> str:
  if not entity:
    return name
  return f"{entity}/{name}"


@dataclass
class BuiltinSensorCfg(SensorCfg):
  sensor_type: SensorType
  objtype: ObjRefType | None = None
  objname: str | None = None
  obj_entity: str | None = None
  reftype: ObjRefType | None = None
  refname: str | None = None
  ref_entity: str | None = None

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
    self._data_slice: slice | None = None

  @classmethod
  def from_existing(cls, name: str) -> BuiltinSensor:
    """Wrap an existing sensor already defined in entity XML."""
    return cls(cfg=None, name=name)

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del entities
    if self.cfg is None:
      return

    for sensor in scene_spec.sensors:
      if sensor.name == self.cfg.name:
        raise ValueError(
          f"Sensor '{self.cfg.name}' already exists (likely defined in entity XML). "
          "Remove the BuiltinSensorCfg to use the XML sensor, or rename one of them."
        )

    if (self.cfg.reftype is None) ^ (self.cfg.refname is None):
      raise ValueError(
        f"Provide both reftype and refname (or neither) for sensor '{self.cfg.name}'."
      )
    if (self.cfg.objtype is None) ^ (self.cfg.objname is None):
      raise ValueError(
        f"Provide both objtype and objname (or neither) for sensor '{self.cfg.name}'."
      )

    kwargs = {
      "name": self.cfg.name,
      "type": _SENSOR_TYPE_MAP[self.cfg.sensor_type],
    }
    if self.cfg.objtype is not None:
      assert self.cfg.objname is not None
      obj_name = _prefix_name(self.cfg.objname, self.cfg.obj_entity)
      kwargs["objtype"] = _OBJECT_TYPE_MAP[self.cfg.objtype]
      kwargs["objname"] = obj_name
    if self.cfg.reftype is not None:
      assert self.cfg.refname is not None
      ref_name = _prefix_name(self.cfg.refname, self.cfg.ref_entity)
      kwargs["reftype"] = _OBJECT_TYPE_MAP[self.cfg.reftype]
      kwargs["refname"] = ref_name

    scene_spec.add_sensor(**kwargs)

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    del model, device
    self._data = data
    sensor = mj_model.sensor(self._name)
    start = sensor.adr[0]
    dim = sensor.dim[0]
    self._data_slice = slice(start, start + dim)

  @property
  def data(self) -> torch.Tensor:
    if self._data is None or self._data_slice is None:
      raise RuntimeError(f"Sensor '{self._name}' not initialized.")
    return self._data.sensordata[:, self._data_slice]
