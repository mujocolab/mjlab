# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Curriculum manager for updating environment quantities subject to a training curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBase
from mjlab.managers.manager_term_config import CurriculumTermCfg
from mjlab.utils.dataclasses import get_terms

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class CurriculumManager(ManagerBase):
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: object, env: ManagerBasedRlEnv):
    self._term_names: list[str] = list()
    self._term_cfgs: list[CurriculumTermCfg] = list()
    self._class_term_cfgs: list[CurriculumTermCfg] = list()

    self.cfg = cfg
    super().__init__(env)

    self._curriculum_state = dict()
    for term_name in self._term_names:
      self._curriculum_state[term_name] = None

  def __str__(self) -> str:
    msg = f"<CurriculumManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Curriculum Terms"
    table.field_names = ["Index", "Name"]
    table.align["Name"] = "l"
    for index, name in enumerate(self._term_names):
      table.add_row([index, name])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    for term_name, term_state in self._curriculum_state.items():
      if term_state is not None:
        data = []
        if isinstance(term_state, dict):
          for _key, value in term_state.items():
            if isinstance(value, torch.Tensor):
              value = value.item()
            terms[term_name].append(value)
        else:
          if isinstance(term_state, torch.Tensor):
            term_state = term_state.item()
          data.append(term_state)
        terms.append((term_name, data))
    return terms

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    extras = {}
    for term_name, term_state in self._curriculum_state.items():
      if term_state is not None:
        if isinstance(term_state, dict):
          for key, value in term_state.items():
            if isinstance(value, torch.Tensor):
              value = value.item()
            extras[f"Curriculum/{term_name}/{key}"] = value
        else:
          if isinstance(term_state, torch.Tensor):
            term_state = term_state.item()
          extras[f"Curriculum/{term_name}"] = term_state
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return extras

  def compute(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=False):
      state = term_cfg.func(self._env, env_ids, **term_cfg.params)
      self._curriculum_state[name] = state

  def _prepare_terms(self):
    cfg_items = get_terms(self.cfg, CurriculumTermCfg).items()
    for term_name, term_cfg in cfg_items:
      term_cfg: CurriculumTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if isinstance(term_cfg.func, ManagerTermBase):
        self._class_term_cfgs.append(term_cfg)


class NullCurriculumManager:
  """Placeholder for absent curriculum manager that safely no-ops all operations."""

  def __init__(self):
    self.active_terms: list[str] = []
    self._curriculum_state: dict[str, Any] = {}
    self.cfg = None

  def __str__(self) -> str:
    return "<NullCurriculumManager> (inactive)"

  def __repr__(self) -> str:
    return "NullCurriculumManager()"

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    return []

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
    return {}

  def compute(self, env_ids: torch.Tensor | None = None) -> None:
    pass
