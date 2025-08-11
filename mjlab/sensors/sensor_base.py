from __future__ import annotations

import abc
import torch
import mujoco
import mujoco_warp as mjwarp
from typing import TYPE_CHECKING, Any, Sequence
from mjlab.entities.indexing import EntityIndexing

if TYPE_CHECKING:
  from mjlab.sensors.sensor_base_config import SensorBaseCfg


class SensorBase(abc.ABC):
  def __init__(self, cfg: SensorBaseCfg):
    if cfg.history_length < 0:
      raise ValueError(f"History length must be >= 0, got {cfg.history_length}")
    self.cfg = cfg

  def initialize(
    self,
    dt: float,
    indexing: EntityIndexing,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    del indexing, mj_model, model  # Unused.

    self._device = device
    self._sim_physics_dt = dt
    self._num_envs = data.nworld

    self._is_outdated = torch.ones(
      self._num_envs, dtype=torch.bool, device=self._device
    )
    self._timestamp = torch.zeros(self._num_envs, device=self._device)
    self._timestamp_last_update = torch.zeros_like(self._timestamp)

  def reset(self, env_ids: Sequence[int] | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._timestamp[env_ids] = 0.0
    self._timestamp_last_update[env_ids] = 0.0
    self._is_outdated[env_ids] = True

  def update(self, dt: float, force_recompute: bool = False) -> None:
    self._timestamp += dt
    self._is_outdated |= (
      self._timestamp - self._timestamp_last_update + 1e-6 >= self.cfg.update_period
    )
    if force_recompute or (self.cfg.history_length > 0):
      self._update_outdated_buffers()

  def _update_outdated_buffers(self):
    outdated_env_ids = self._is_outdated.nonzero().squeeze(-1)
    if len(outdated_env_ids) > 0:
      self._update_buffers_impl(outdated_env_ids)
      self._timestamp_last_update[outdated_env_ids] = self._timestamp[outdated_env_ids]
      self._is_outdated[outdated_env_ids] = False

  @property
  def device(self) -> str:
    return self._device

  @property
  def num_instances(self) -> int:
    return self._num_envs

  @property
  @abc.abstractmethod
  def data(self) -> Any:
    raise NotImplementedError

  @abc.abstractmethod
  def _update_buffers_impl(self, env_ids: Sequence[int]):
    raise NotImplementedError
