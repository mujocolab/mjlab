"""Observation manager for computing observations."""

from typing import Sequence

import numpy as np
import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.dataclasses import get_terms
from mjlab.utils.noise import noise_cfg, noise_model


class ObservationManager(ManagerBase):
  def __init__(self, cfg: object, env):
    self.cfg = cfg
    super().__init__(env=env)

    self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()

    for group_name, group_term_dims in self._group_obs_term_dim.items():
      if self._group_obs_concatenate[group_name]:
        # All observation terms will be concatenated.
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
        # Observation terms will be returned as a dictionary; we will store a
        # list of their dimensions.
        self._group_obs_dim[group_name] = group_term_dims

    self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None

  def __str__(self) -> str:
    msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"
    for group_name, group_dim in self._group_obs_dim.items():
      table = PrettyTable()
      table.title = f"Active Observation Terms in Group: '{group_name}'"
      if self._group_obs_concatenate[group_name]:
        table.title += f" (shape: {group_dim})"  # type: ignore
      table.field_names = ["Index", "Name", "Shape"]
      table.align["Name"] = "l"
      obs_terms = zip(
        self._group_obs_term_names[group_name],
        self._group_obs_term_dim[group_name],
        strict=False,
      )
      for index, (name, dims) in enumerate(obs_terms):
        tab_dims = tuple(dims)
        table.add_row([index, name, tab_dims])
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
    for _group_name, group_cfg in self._group_obs_class_term_cfgs.items():
      for term_cfg in group_cfg:
        term_cfg.func.reset(env_ids=env_ids)
      # TODO: History.
    for mod in self._group_obs_class_instances.values():
      mod.reset(env_ids=env_ids)
    return {}

  def compute(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = dict()
    for group_name in self._group_obs_term_names:
      obs_buffer[group_name] = self.compute_group(group_name)
    self._obs_buffer = obs_buffer
    return obs_buffer

  def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
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

    group_cfg_items = get_terms(self.cfg, ObservationGroupCfg).items()
    for group_name, group_cfg in group_cfg_items:
      if group_cfg is None:
        print(f"group: {group_name} set to None, skipping...")
        continue
      group_cfg: ObservationGroupCfg

      self._group_obs_term_names[group_name] = list()
      self._group_obs_term_dim[group_name] = list()
      self._group_obs_term_cfgs[group_name] = list()
      self._group_obs_class_term_cfgs[group_name] = list()

      self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
      self._group_obs_concatenate_dim[group_name] = (
        group_cfg.concatenate_dim + 1
        if group_cfg.concatenate_dim >= 0
        else group_cfg.concatenate_dim
      )

      group_cfg_items = get_terms(group_cfg, ObservationTermCfg).items()
      for term_name, term_cfg in group_cfg_items:
        if term_cfg is None:
          print(f"term: {term_name} set to None, skipping...")
          continue

        self._resolve_common_term_cfg(term_name, term_cfg)

        if not group_cfg.enable_corruption:
          term_cfg.noise = None
        self._group_obs_term_names[group_name].append(term_name)
        self._group_obs_term_cfgs[group_name].append(term_cfg)

        obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)
        self._group_obs_term_dim[group_name].append(obs_dims[1:])

        # Prepare noise model classes.
        if term_cfg.noise is not None and isinstance(
          term_cfg.noise, noise_cfg.NoiseModelCfg
        ):
          noise_model_cls = term_cfg.noise.class_type
          assert issubclass(noise_model_cls, noise_model.NoiseModel), (
            f"Class type for observation term '{term_name}' NoiseModelCfg"
            f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
          )
          # Create and store noise model instance.
          self._group_obs_class_instances[term_name] = noise_model_cls(
            term_cfg.noise, num_envs=self._env.num_envs, device=self._env.device
          )
