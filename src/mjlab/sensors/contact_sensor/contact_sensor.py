from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entities import entity
from mjlab.entities.indexing import EntityIndexing
from mjlab.sensors.sensor_base import SensorBase
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from mjlab.utils.spec import construct_contact_sensor_intprm

if TYPE_CHECKING:
  from mjlab.sensors.contact_sensor.contact_sensor_config import ContactSensorCfg


@dataclass
class ContactSensorData:
  net_forces_w: torch.Tensor
  net_forces_w_history: torch.Tensor
  last_air_time: torch.Tensor | None
  current_air_time: torch.Tensor | None
  last_contact_time: torch.Tensor | None
  current_contact_time: torch.Tensor | None


class ContactSensor(SensorBase):
  cfg: ContactSensorCfg

  def __init__(self, cfg: ContactSensorCfg):
    super().__init__(cfg)

    self._data: ContactSensorData | None = None

  def edit_spec(self, entity: entity.Entity, spec: mujoco.MjSpec) -> None:
    all_body_names = [b.name for b in entity.spec.bodies if b.name != "world"]
    filter_expr = [f"{self.cfg.entity_name}/{f}" for f in self.cfg.filter_expr]
    self._body_names = resolve_matching_names(filter_expr, all_body_names)[1]
    self._sensor_names: list[str] = []
    for body_name in self._body_names:
      sns = spec.add_sensor(
        name=f"contact_sensor/{body_name}",
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname=body_name,
      )
      sns.intprm[0:3] = construct_contact_sensor_intprm(
        data="found force", reduce="netforce"
      )
      self._sensor_names.append(sns.name)

  @property
  def data(self) -> ContactSensorData:
    self._update_outdated_buffers()
    assert self._data is not None
    return self._data

  @property
  def num_bodies(self) -> int:
    return len(self.body_names)

  @property
  def body_names(self) -> list[str]:
    return self._body_names

  def find_bodies(
    self, name_keys: str | Sequence[str], preserve_order: bool = False
  ) -> tuple[list[int], list[str]]:
    prefix = f"{self.cfg.entity_name}/"
    stripped_body_names = [
      name.removeprefix(prefix) if name.startswith(prefix) else name
      for name in self.body_names
    ]
    ids, stripped_names = resolve_matching_names(
      name_keys, stripped_body_names, preserve_order
    )
    names = [f"{prefix}{name}" for name in stripped_names]
    return ids, names

  def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
    if not self.cfg.track_air_time:
      raise RuntimeError()
    assert self.data.current_contact_time is not None
    assert self.data.current_air_time is not None
    currently_in_contact = self.data.current_contact_time > 0.0
    less_than_dt_in_contact = self.data.current_contact_time < (dt + abs_tol)
    return currently_in_contact * less_than_dt_in_contact

  def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
    if not self.cfg.track_air_time:
      raise RuntimeError()
    assert self.data.current_air_time is not None
    currently_detached = self.data.current_air_time > 0.0
    less_than_dt_detached = self.data.current_air_time < (dt + abs_tol)
    return currently_detached * less_than_dt_detached

  def initialize(
    self,
    dt: float,
    indexing: EntityIndexing,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(dt, indexing, mj_model, model, data, device)
    self._wp_data = data
    self._wp_model = model

    self._sns_start_indices = []
    self._sns_end_indices = []
    for name in self._sensor_names:
      sns = mj_model.sensor(name)
      start_idx = int(sns.adr[0])
      dim = int(sns.dim[0])
      self._sns_start_indices.append(start_idx)
      self._sns_end_indices.append(start_idx + dim)
    self._sns_start_indices = torch.tensor(self._sns_start_indices, device=self._device)
    self._sns_end_indices = torch.tensor(self._sns_end_indices, device=self._device)

    # Prepare data buffers.
    net_forces_w = torch.zeros(self._num_envs, self.num_bodies, 3, device=self._device)
    if self.cfg.history_length > 0:
      net_forces_w_history = torch.zeros(
        self._num_envs,
        self.cfg.history_length,
        self.num_bodies,
        3,
        device=self._device,
      )
    else:
      net_forces_w_history = net_forces_w.unsqueeze(1)
    if self.cfg.track_air_time:
      self._data = ContactSensorData(
        net_forces_w=net_forces_w,
        net_forces_w_history=net_forces_w_history,
        last_air_time=torch.zeros(self._num_envs, self.num_bodies, device=self._device),
        current_air_time=torch.zeros(
          self._num_envs, self.num_bodies, device=self._device
        ),
        last_contact_time=torch.zeros(
          self._num_envs, self.num_bodies, device=self._device
        ),
        current_contact_time=torch.zeros(
          self._num_envs, self.num_bodies, device=self._device
        ),
      )
    else:
      self._data = ContactSensorData(
        net_forces_w=net_forces_w,
        net_forces_w_history=net_forces_w_history,
        last_air_time=None,
        current_air_time=None,
        last_contact_time=None,
        current_contact_time=None,
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)

    assert self._data is not None
    self._data.net_forces_w[env_ids] = 0.0
    self._data.net_forces_w_history[env_ids] = 0.0
    if self.cfg.track_air_time:
      assert self._data.current_air_time is not None
      assert self._data.last_air_time is not None
      assert self._data.current_contact_time is not None
      assert self._data.last_contact_time is not None
      self._data.current_air_time[env_ids] = 0.0
      self._data.last_air_time[env_ids] = 0.0
      self._data.current_contact_time[env_ids] = 0.0
      self._data.last_contact_time[env_ids] = 0.0

  def _update_buffers_impl(self, env_ids: torch.Tensor | slice | None = None):
    """Update contact force buffers for specified environments."""
    if env_ids is None:
      env_ids = slice(None)
    assert self._data is not None
    self._data.net_forces_w[env_ids] = 0.0
    self._compute_contact_forces(env_ids)
    self._update_force_history(env_ids)
    if self.cfg.track_air_time:
      self._update_contact_timing(env_ids)

  def _compute_contact_forces(self, env_ids):
    """Compute net contact forces for each body."""
    assert self._data is not None
    sensordata = self._wp_data.sensordata[env_ids].clone()
    for body_idx, (start_idx, end_idx) in enumerate(
      zip(self._sns_start_indices, self._sns_end_indices, strict=True)
    ):
      sensor_values = sensordata[:, start_idx:end_idx]
      self._data.net_forces_w[env_ids, body_idx] = sensor_values[:, 1:4]

  def _update_force_history(self, env_ids):
    """Update force history buffer if configured."""
    assert self._data is not None
    if self.cfg.history_length > 0:
      history = self._data.net_forces_w_history[env_ids]
      self._data.net_forces_w_history[env_ids] = history.roll(1, dims=1)
      self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]

  def _update_contact_timing(self, env_ids):
    """Track air and contact time for each body."""
    assert self._data is not None
    elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
    elapsed_time = elapsed_time.unsqueeze(-1)

    # Determine contact state.
    force_magnitudes = torch.norm(self._data.net_forces_w[env_ids], dim=-1)
    is_contact = force_magnitudes > self.cfg.force_threshold

    # Detect state transitions.
    assert self._data.current_air_time is not None
    assert self._data.current_contact_time is not None
    is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact
    is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact

    self._update_air_time(env_ids, elapsed_time, is_contact, is_first_contact)
    self._update_contact_time(env_ids, elapsed_time, is_contact, is_first_detached)

  def _update_air_time(self, env_ids, elapsed_time, is_contact, is_first_contact):
    """Update air time counters."""
    # Save total air time when first making contact.
    assert self._data is not None
    assert self._data.current_air_time is not None
    assert self._data.last_air_time is not None
    self._data.last_air_time[env_ids] = torch.where(
      is_first_contact,
      self._data.current_air_time[env_ids] + elapsed_time,
      self._data.last_air_time[env_ids],
    )

    # Increment current air time for bodies not in contact.
    self._data.current_air_time[env_ids] = torch.where(
      ~is_contact,
      self._data.current_air_time[env_ids] + elapsed_time,
      0.0,
    )

  def _update_contact_time(self, env_ids, elapsed_time, is_contact, is_first_detached):
    """Update contact time counters."""
    # Save total contact time when first detaching.
    assert self._data is not None
    assert self._data.current_contact_time is not None
    assert self._data.last_contact_time is not None
    self._data.last_contact_time[env_ids] = torch.where(
      is_first_detached,
      self._data.current_contact_time[env_ids] + elapsed_time,
      self._data.last_contact_time[env_ids],
    )

    # Increment current contact time for bodies in contact.
    self._data.current_contact_time[env_ids] = torch.where(
      is_contact,
      self._data.current_contact_time[env_ids] + elapsed_time,
      0.0,
    )
