from typing import Any, Sequence

import numpy as np
import torch
from mjlab.managers.manager_base import ManagerBase
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg

from mjlab.utils.dataclasses import get_terms


class ObservationManager(ManagerBase):
  def __init__(self, cfg: object, env):
    super().__init__(cfg=cfg, env=env)

    self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None

  # Properties.

  @property
  def active_terms(self) -> list[str] | dict[str, list[str]]:
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

  def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
    for group_name, group_cfg in self._group_obs_class_term_cfgs.items():
      for term_cfg in group_cfg:
        term_cfg.func.reset(env_ids=env_ids)
    for mod in self._group_obs_class_instances:
      mod.reset(env_ids=env_ids)
    return {}

  def compute(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    obs_buffer = dict()
    for group_name in self._group_obs_term_names:
      obs_buffer[group_name] = self.compute_group(group_name)
    self._obs_buffer = obs_buffer
    return obs_buffer

  def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
    group_term_names = self._group_obs_term_names[group_name]
    group_obs = dict.fromkeys(group_term_names, None)
    obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])
    for term_name, term_cfg in obs_terms:
      obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
      group_obs[term_name] = obs
    if self._group_obs_concatenate[group_name]:
      return torch.cat(
        list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name]
      )
    return group_obs

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []

    if self._obs_buffer is None:
      self.compute()
    obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer

    for group_name, _ in self.group_obs_dim.items():
      if not self.group_obs_concatenate[group_name]:
        for name, term in obs_buffer[group_name].items():
          terms.append((group_name + "-" + name, term[env_idx].cpu().tolist()))
        continue

      idx = 0
      data = obs_buffer[group_name]
      for name, shape in zip(
        self._group_obs_term_names[group_name],
        self._group_obs_term_dim[group_name],
      ):
        data_length = np.prod(shape)
        term = data[env_idx, idx : idx + data_length]
        terms.append((group_name + "-" + name, term.cpu().tolist()))
        idx += data_length

    return terms

  def _prepare_terms(self) -> None:
    self._group_obs_term_names: dict[str, list[str]] = dict()
    self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
    self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
    self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
    self._group_obs_concatenate: dict[str, bool] = dict()
    self._group_obs_concatenate_dim: dict[str, int] = dict()
    self._group_obs_class_instances: list[Any] = list()

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

        # TODO: resolve
        self._resolve_common_term_cfg(term_name, term_cfg)

        self._group_obs_term_names[group_name].append(term_name)
        self._group_obs_term_cfgs[group_name].append(term_cfg)

        obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)
        self._group_obs_term_dim[group_name].append(obs_dims[1:])

        # if isinstance(term_cfg.func, ManagerTermBase):
        #   self._group_obs_class_term_cfgs[group_name].append(term_cfg)
        #   term_cfg.func.reset()
