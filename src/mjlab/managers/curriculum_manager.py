from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

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

    super().__init__(cfg, env)

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
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if isinstance(term_cfg.func, ManagerTermBase):
        self._class_term_cfgs.append(term_cfg)
