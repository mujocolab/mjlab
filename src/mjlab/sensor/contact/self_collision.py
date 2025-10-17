from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.sensor.base import Sensor, SensorCfg
from mjlab.sensor.contact.core import ContactSensor, ContactSensorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass
class SelfCollisionSensorCfg(SensorCfg):
  """Convenience wrapper that builds a self-collision `ContactSensor`."""

  name: str
  entity_name: str

  def build(self) -> Sensor:
    return SelfCollisionSensor(self)


class SelfCollisionSensor(Sensor):
  """Self-collision sensor built on top of `ContactSensor`."""

  def __init__(self, cfg: SelfCollisionSensorCfg) -> None:
    super().__init__(cfg)
    self.name = cfg.name
    self._cfg = cfg
    self._contact_sensor: ContactSensor | None = None

  def edit_spec(
    self,
    scene_spec: mujoco.MjSpec,
    entities: dict[str, Entity],
  ) -> None:
    entity = entities.get(self._cfg.entity_name)
    if entity is None:
      raise ValueError(
        f"Entity '{self._cfg.entity_name}' not found in scene. "
        f"Available: {list(entities.keys())}"
      )

    root_body = entity.body_names[0]
    contact_cfg = ContactSensorCfg(
      name=self.name,
      entity_name=self._cfg.entity_name,
      subtree1=root_body,
      subtree2=root_body,
      reduce="netforce",
      data=("found",),
    )
    self._contact_sensor = ContactSensor(contact_cfg)
    self._contact_sensor.edit_spec(scene_spec, entities)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    assert self._contact_sensor is not None
    self._contact_sensor.initialize(mj_model, model, data, device)

  @property
  def data(self) -> torch.Tensor:
    assert self._contact_sensor is not None
    count = self._contact_sensor.count
    return count.squeeze(-1) if count.dim() > 1 else count

  @property
  def count(self) -> torch.Tensor:
    return self.data

  def update(self, dt: float) -> None:
    if self._contact_sensor is not None:
      self._contact_sensor.update(dt)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._contact_sensor is not None:
      self._contact_sensor.reset(env_ids)
