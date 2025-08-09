from __future__ import annotations

import torch
import mujoco
import warp as wp
import mujoco_warp as mjwarp
from mjlab.entities.indexing import EntityIndexing
from typing import TYPE_CHECKING, Sequence
from mjlab.sensors.sensor_base import SensorBase
from mjlab.sensors.contact_sensor.contact_sensor_data import ContactSensorData
from mjlab.utils import string as string_utils

if TYPE_CHECKING:
  from mjlab.sensors.contact_sensor.contact_sensor_config import ContactSensorCfg


class ContactSensor(SensorBase):
  cfg: ContactSensorCfg

  def __init__(self, cfg: ContactSensorCfg):
    super().__init__(cfg)

    self._data = ContactSensorData()

  @property
  def data(self) -> ContactSensorData:
    self._update_outdated_buffers()
    return self._data

  def find_bodies(
    self, name_keys: str | Sequence[str], preserve_order: bool = False
  ) -> tuple[list[int], list[str]]:
    prefix = f"{self.cfg.entity_name}/"
    stripped_body_names = [
      name.removeprefix(prefix) if name.startswith(prefix) else name
      for name in self.body_names
    ]
    ids, stripped_names = string_utils.resolve_matching_names(
      name_keys, stripped_body_names, preserve_order
    )
    names = [f"{prefix}{name}" for name in stripped_names]
    return ids, names

  def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
    if not self.cfg.track_air_time:
      raise RuntimeError()
    currently_in_contact = self.data.current_contact_time > 0.0
    less_than_dt_in_contact = self.data.current_contact_time < (dt + abs_tol)
    return currently_in_contact * less_than_dt_in_contact

  def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
    if not self.cfg.track_air_time:
      raise RuntimeError()
    currently_detached = self.data.current_air_time > 0.0
    less_than_dt_detached = self.data.current_air_time < (dt + abs_tol)
    return currently_detached * less_than_dt_detached

  def initialize(
    self,
    indexing: EntityIndexing,
    model: mujoco.MjModel,
    data: mjwarp.Data,
    device: str,
    wp_model: mjwarp.Model,
  ) -> None:
    super().initialize(indexing, model, data, device, wp_model)
    self._wp_data = data
    self._wp_model = wp_model

    all_body_names = [model.body(i).name for i in indexing.body_ids]
    filter_expr = [f"{self.cfg.entity_name}/{f}" for f in self.cfg.filter_expr]
    self._body_names = string_utils.resolve_matching_names(filter_expr, all_body_names)[
      1
    ]
    self._body_ids = [model.body(n).id for n in self._body_names]
    self._num_bodies = len(self._body_names)

    all_geom_names = [model.geom(i).name for i in indexing.geom_ids]
    geom_filter_expr = [
      f"{self.cfg.entity_name}/{f}" for f in self.cfg.geom_filter_expr
    ]
    subset_geom_names = string_utils.resolve_matching_names(
      geom_filter_expr, all_geom_names
    )[1]
    subset_geom_ids = [model.geom(n).id for n in subset_geom_names]

    self._body2geom = {}
    all_geom_ids = []
    geom_to_body_map = []
    for i, bid in enumerate(self._body_ids):
      geom_ids = []
      for geom_id in range(model.ngeom):
        if model.geom(geom_id).bodyid == bid and geom_id in subset_geom_ids:
          geom_ids.append(geom_id)
      all_geom_ids.extend(geom_ids)
      geom_to_body_map.extend([i] * len(geom_ids))
      self._body2geom[bid] = torch.tensor(geom_ids, dtype=torch.int32, device=device)
    self.all_geom_ids = torch.tensor(all_geom_ids, device=device)
    self.geom_to_body_map = torch.tensor(geom_to_body_map, device=device)

    # Prepare data buffers.
    self._data.net_forces_w = torch.zeros(
      self._num_envs, self._num_bodies, 3, device=self._device
    )
    if self.cfg.history_length > 0:
      self._data.net_forces_w_history = torch.zeros(
        self._num_envs,
        self.cfg.history_length,
        self._num_bodies,
        3,
        device=self._device,
      )
    else:
      self._data.net_forces_w_history = self._data.net_forces_w.unsqueeze(1)
    if self.cfg.track_air_time:
      self._data.last_air_time = torch.zeros(
        self._num_envs, self._num_bodies, device=self._device
      )
      self._data.current_air_time = torch.zeros(
        self._num_envs, self._num_bodies, device=self._device
      )
      self._data.last_contact_time = torch.zeros(
        self._num_envs, self._num_bodies, device=self._device
      )
      self._data.current_contact_time = torch.zeros(
        self._num_envs, self._num_bodies, device=self._device
      )

  def reset(self, env_ids: Sequence[int] | None = None):
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._data.net_forces_w[env_ids] = 0.0
    self._data.net_forces_w_history[env_ids] = 0.0
    if self.cfg.track_air_time:
      self._data.current_air_time[env_ids] = 0.0
      self._data.last_air_time[env_ids] = 0.0
      self._data.current_contact_time[env_ids] = 0.0
      self._data.last_contact_time[env_ids] = 0.0

  def _update_buffers_impl(self, env_ids: Sequence[int]):
    if len(env_ids) == self._num_envs:
      env_ids = slice(None)

    contact = self._wp_data.contact

    masks = (contact.geom[:, 0][None, :] == self.all_geom_ids[:, None]) | (
      contact.geom[:, 1][None, :] == self.all_geom_ids[:, None]
    )

    has_contacts = torch.any(masks, dim=1)
    distances = torch.where(masks, contact.dist[None, :], 1e4)
    contact_indices = distances.argmin(dim=1)

    valid_geom_mask = has_contacts
    valid_contact_indices = contact_indices[valid_geom_mask]
    valid_body_indices = self.geom_to_body_map[valid_geom_mask]

    if len(valid_contact_indices) > 0:
      force = wp.zeros(len(valid_contact_indices), dtype=wp.spatial_vector)
      contact_ids = wp.from_torch(valid_contact_indices.int(), dtype=wp.int32)

      mjwarp.contact_force(
        m=self._wp_model,
        d=self._wp_data.struct,
        contact_ids=contact_ids,
        to_world_frame=True,
        force=force,
      )

      forces_torch = wp.to_torch(force)[:, :3]
      frames = contact.frame[valid_contact_indices]
      normals = frames[:, 0]

      normal_force_magnitudes = torch.sum(
        forces_torch * normals, dim=1, keepdim=True
      )  # [num_valid_contacts, 1]
      normal_forces = normal_force_magnitudes * normals

      body_forces = torch.zeros(
        (len(self._body_ids), 3), device=normal_forces.device, dtype=normal_forces.dtype
      )
      body_forces.scatter_add_(
        0, valid_body_indices.unsqueeze(1).expand(-1, 3), normal_forces
      )

      self._data.net_forces_w[env_ids] += body_forces

    if self.cfg.history_length > 0:
      self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[
        env_ids
      ].roll(1, dims=1)
      self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]

    if self.cfg.track_air_time:
      elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
      is_contact = (
        torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1)
        > self.cfg.force_threshold
      )
      is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact
      is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact
      self._data.last_air_time[env_ids] = torch.where(
        is_first_contact,
        self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
        self._data.last_air_time[env_ids],
      )
      # -- increment time for bodies that are not in contact
      self._data.current_air_time[env_ids] = torch.where(
        ~is_contact,
        self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
        0.0,
      )
      # -- update the last contact time if body has just detached
      self._data.last_contact_time[env_ids] = torch.where(
        is_first_detached,
        self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
        self._data.last_contact_time[env_ids],
      )
      # -- increment time for bodies that are in contact
      self._data.current_contact_time[env_ids] = torch.where(
        is_contact,
        self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
        0.0,
      )

  @property
  def num_bodies(self) -> int:
    return self._num_bodies

  @property
  def body_names(self) -> list[str]:
    return self._body_names
