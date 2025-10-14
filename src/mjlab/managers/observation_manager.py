"""Observation manager for computing observations."""

from typing import Sequence

import numpy as np
import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.buffers import CircularBuffer
from mjlab.utils.dataclasses import get_terms
from mjlab.utils.noise import noise_cfg, noise_model


class ObservationManager(ManagerBase):
  def __init__(self, cfg: object, env):
    self.cfg = cfg
    super().__init__(env=env)

    self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()

    for group_name, group_term_dims in self._group_obs_term_dim.items():
      if self._group_obs_concatenate[group_name]:
        try:
          term_dims = torch.stack(
            [torch.tensor(dims, device="cpu") for dims in group_term_dims], dim=0
          )
          if len(term_dims.shape) > 1:
            if self._group_obs_concatenate_dim[group_name] >= 0:
              dim = self._group_obs_concatenate_dim[group_name] - 1
            else:
              dim = self._group_obs_concatenate_dim[group_name]
            dim_sum = torch.sum(term_dims[:, dim], dim=0)
            term_dims[0, dim] = dim_sum
            term_dims = term_dims[0]
          else:
            term_dims = torch.sum(term_dims, dim=0)
          self._group_obs_dim[group_name] = tuple(term_dims.tolist())
        except RuntimeError:
          raise RuntimeError(
            f"Unable to concatenate observation terms in group {group_name}."
          ) from None
      else:
        self._group_obs_dim[group_name] = group_term_dims

    self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None

  def __str__(self) -> str:
    msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"
    for group_name, group_dim in self._group_obs_dim.items():
      table = PrettyTable()
      table.title = f"Active Observation Terms in Group: '{group_name}'"
      if self._group_obs_concatenate[group_name]:
        table.title += f" (shape: {group_dim})"  # type: ignore
      table.field_names = ["Index", "Name", "Shape", "History"]
      table.align["Name"] = "l"
      obs_terms = zip(
        self._group_obs_term_names[group_name],
        self._group_obs_term_dim[group_name],
        self._group_obs_term_cfgs[group_name],
        strict=False,
      )
      for index, (name, dims, term_cfg) in enumerate(obs_terms):
        tab_dims = tuple(dims)
        if term_cfg.history_length > 0:
          history_str = f"{term_cfg.history_length}"
          if term_cfg.flatten_history_dim:
            history_str += " (flat)"
        else:
          history_str = "-"
        table.add_row([index, name, tab_dims, history_str])
      msg += table.get_string()
      msg += "\n"
    return msg

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []

    if self._obs_buffer is None:
      self.compute()
    assert self._obs_buffer is not None
    obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer

    for group_name, _ in self.group_obs_dim.items():
      if not self.group_obs_concatenate[group_name]:
        buffers = obs_buffer[group_name]
        assert isinstance(buffers, dict)
        for name, term in buffers.items():
          terms.append((group_name + "-" + name, term[env_idx].cpu().tolist()))
        continue

      idx = 0
      data = obs_buffer[group_name]
      assert isinstance(data, torch.Tensor)
      for name, shape in zip(
        self._group_obs_term_names[group_name],
        self._group_obs_term_dim[group_name],
        strict=False,
      ):
        data_length = np.prod(shape)
        term = data[env_idx, idx : idx + data_length]
        terms.append((group_name + "-" + name, term.cpu().tolist()))
        idx += data_length

    return terms

  # Properties.

  @property
  def active_terms(self) -> dict[str, list[str]]:
    return self._group_obs_term_names

  @property
  def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
    return self._group_obs_dim

  @property
  def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
    return self._group_obs_term_dim

  @property
  def group_obs_concatenate(self) -> dict[str, bool]:
    return self._group_obs_concatenate

  # Methods.

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    for group_name, group_cfg in self._group_obs_class_term_cfgs.items():
      for term_cfg in group_cfg:
        term_cfg.func.reset(env_ids=env_ids)
      for term_name in self._group_obs_term_names[group_name]:
        if term_name in self._group_obs_term_history_buffer[group_name]:
          batch_ids = None if isinstance(env_ids, slice) else env_ids
          self._group_obs_term_history_buffer[group_name][term_name].reset(
            batch_ids=batch_ids
          )
    for mod in self._group_obs_class_instances.values():
      mod.reset(env_ids=env_ids)
    return {}

  def compute(
    self, update_history: bool = False
  ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = dict()
    for group_name in self._group_obs_term_names:
      obs_buffer[group_name] = self.compute_group(group_name, update_history)
    self._obs_buffer = obs_buffer
    return obs_buffer

  def compute_group(
    self, group_name: str, update_history: bool = False
  ) -> torch.Tensor | dict[str, torch.Tensor]:
    group_term_names = self._group_obs_term_names[group_name]
    group_obs: dict[str, torch.Tensor] = {}
    obs_terms = zip(
      group_term_names, self._group_obs_term_cfgs[group_name], strict=False
    )
    for term_name, term_cfg in obs_terms:
      obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
      if isinstance(term_cfg.noise, noise_cfg.NoiseCfg):
        obs = term_cfg.noise.apply(obs)
      elif isinstance(term_cfg.noise, noise_cfg.NoiseModelCfg):
        obs = self._group_obs_class_instances[term_name](obs)
      if term_cfg.clip:
        obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
      if term_cfg.scale is not None:
        scale = term_cfg.scale
        assert isinstance(scale, torch.Tensor)
        obs = obs.mul_(scale)
      if term_cfg.history_length > 0:
        circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
        if circular_buffer._buffer is None:
          circular_buffer.append(obs)
        if update_history:
          circular_buffer.append(obs)

        if term_cfg.flatten_history_dim:
          group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
        else:
          group_obs[term_name] = circular_buffer.buffer
      else:
        group_obs[term_name] = obs
    if self._group_obs_concatenate[group_name]:
      return torch.cat(
        list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name]
      )
    return group_obs

  def _prepare_terms(self) -> None:
    self._group_obs_term_names: dict[str, list[str]] = dict()
    self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
    self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
    self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
    self._group_obs_concatenate: dict[str, bool] = dict()
    self._group_obs_concatenate_dim: dict[str, int] = dict()
    self._group_obs_class_instances: dict[str, noise_model.NoiseModel] = {}
    self._group_obs_term_history_buffer: dict[str, dict[str, CircularBuffer]] = dict()

    group_cfg_items = get_terms(self.cfg, ObservationGroupCfg).items()
    for group_name, group_cfg in group_cfg_items:
      group_cfg: ObservationGroupCfg | None
      if group_cfg is None:
        print(f"group: {group_name} set to None, skipping...")
        continue

      self._group_obs_term_names[group_name] = list()
      self._group_obs_term_dim[group_name] = list()
      self._group_obs_term_cfgs[group_name] = list()
      self._group_obs_class_term_cfgs[group_name] = list()
      group_entry_history_buffer: dict[str, CircularBuffer] = dict()

      self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
      self._group_obs_concatenate_dim[group_name] = (
        group_cfg.concatenate_dim + 1
        if group_cfg.concatenate_dim >= 0
        else group_cfg.concatenate_dim
      )

      group_cfg_items = get_terms(group_cfg, ObservationTermCfg).items()
      for term_name, term_cfg in group_cfg_items:
        term_cfg: ObservationTermCfg | None
        if term_cfg is None:
          print(f"term: {term_name} set to None, skipping...")
          continue

        self._resolve_common_term_cfg(term_name, term_cfg)

        if not group_cfg.enable_corruption:
          term_cfg.noise = None
        if group_cfg.history_length is not None:
          term_cfg.history_length = group_cfg.history_length
          term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
        self._group_obs_term_names[group_name].append(term_name)
        self._group_obs_term_cfgs[group_name].append(term_cfg)
        if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
          self._group_obs_class_term_cfgs[group_name].append(term_cfg)

        obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)

        if term_cfg.scale is not None:
          term_cfg.scale = torch.tensor(
            term_cfg.scale, dtype=torch.float, device=self._env.device
          )

        if term_cfg.noise is not None and isinstance(
          term_cfg.noise, noise_cfg.NoiseModelCfg
        ):
          noise_model_cls = term_cfg.noise.class_type
          assert issubclass(noise_model_cls, noise_model.NoiseModel), (
            f"Class type for observation term '{term_name}' NoiseModelCfg"
            f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
          )
          self._group_obs_class_instances[term_name] = noise_model_cls(
            term_cfg.noise, num_envs=self._env.num_envs, device=self._env.device
          )

        if term_cfg.history_length > 0:
          group_entry_history_buffer[term_name] = CircularBuffer(
            max_len=term_cfg.history_length,
            batch_size=self._env.num_envs,
            device=self._env.device,
          )
          old_dims = list(obs_dims)
          old_dims.insert(1, term_cfg.history_length)
          obs_dims = tuple(old_dims)
          if term_cfg.flatten_history_dim:
            obs_dims = (obs_dims[0], int(np.prod(obs_dims[1:])))

        self._group_obs_term_dim[group_name].append(obs_dims[1:])
      self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer
