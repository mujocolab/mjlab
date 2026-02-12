"""Curriculum manager for updating environment quantities subject to a training curriculum."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class CurriculumTermCfg(ManagerTermBaseCfg):
  """Configuration for a curriculum term.

  Curriculum terms modify environment parameters during training to implement
  curriculum learning strategies (e.g., gradually increasing task difficulty).
  """

  pass


def resolve_curriculum_iterations(
  curriculum: dict[str, CurriculumTermCfg],
  num_steps_per_env: int,
) -> None:
  """Convert ``"iteration"`` keys to ``"step"`` keys in curriculum stages.

  This allows curriculum configs to express thresholds in training
  iterations (more intuitive) rather than raw environment steps.
  The conversion is ``step = iteration * num_steps_per_env``.

  Modifies *curriculum* in-place. Raises :class:`ValueError` if a stage
  dict contains both ``"iteration"`` and ``"step"`` keys.
  """
  for term_cfg in curriculum.values():
    for value in term_cfg.params.values():
      if not isinstance(value, list):
        continue
      for item in value:
        if not isinstance(item, dict):
          continue
        if "iteration" in item and "step" in item:
          raise ValueError(f"Curriculum stage has both 'iteration' and 'step': {item}")
        if "iteration" in item:
          item["step"] = item.pop("iteration") * num_steps_per_env


class CurriculumManager(ManagerBase):
  """Manages curriculum learning for the environment.

  The curriculum manager updates environment parameters during training based
  on agent performance. Each term can modify different aspects of the task
  difficulty (e.g., terrain complexity, command ranges).
  """

  _env: ManagerBasedRlEnv

  def __init__(self, cfg: dict[str, CurriculumTermCfg], env: ManagerBasedRlEnv):
    self._term_names: list[str] = list()
    self._term_cfgs: list[CurriculumTermCfg] = list()
    self._class_term_cfgs: list[CurriculumTermCfg] = list()

    self.cfg = deepcopy(cfg)
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

  # Methods.

  def get_term_cfg(self, term_name: str) -> CurriculumTermCfg:
    if term_name not in self._term_names:
      raise ValueError(f"Term '{term_name}' not found in active terms.")
    return self._term_cfgs[self._term_names.index(term_name)]

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
    for term_name, term_cfg in self.cfg.items():
      term_cfg: CurriculumTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
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
