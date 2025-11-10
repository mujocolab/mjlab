"""Action manager for processing actions sent to the environment."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_term_config import ActionTermCfg


class ActionTerm(ManagerTermBase):
  """Base class for action terms."""

  def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    super().__init__(env)
    self._asset = self._env.scene[self.cfg.asset_name]

  @property
  @abc.abstractmethod
  def action_dim(self) -> int:
    raise NotImplementedError

  @abc.abstractmethod
  def process_actions(self, actions: torch.Tensor) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def apply_actions(self) -> None:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def raw_action(self) -> torch.Tensor:
    raise NotImplementedError


class ActionManager(ManagerBase):
  def __init__(self, cfg: dict[str, ActionTermCfg], env: ManagerBasedRlEnv):
    self.cfg = cfg
    super().__init__(env=env)

    # Create buffers to store actions.
    self._action = torch.zeros(
      (self.num_envs, self.total_action_dim), device=self.device
    )
    self._prev_action = torch.zeros_like(self._action)

  def __str__(self) -> str:
    msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = f"Active Action Terms (shape: {self.total_action_dim})"
    table.field_names = ["Index", "Name", "Dimension"]
    table.align["Name"] = "l"
    table.align["Dimension"] = "r"
    for index, (name, term) in enumerate(self._terms.items()):
      table.add_row([index, name, term.action_dim])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def total_action_dim(self) -> int:
    return sum(self.action_term_dim)

  @property
  def action_term_dim(self) -> list[int]:
    return [term.action_dim for term in self._terms.values()]

  @property
  def action(self) -> torch.Tensor:
    return self._action

  @property
  def prev_action(self) -> torch.Tensor:
    return self._prev_action

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  # Methods.

  def get_term(self, name: str) -> ActionTerm:
    return self._terms[name]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    if env_ids is None:
      env_ids = slice(None)
    # Reset action history.
    self._prev_action[env_ids] = 0.0
    self._action[env_ids] = 0.0
    # Reset action terms.
    for term in self._terms.values():
      term.reset(env_ids=env_ids)
    return {}

  def process_action(self, action: torch.Tensor) -> None:
    if self.total_action_dim != action.shape[1]:
      raise ValueError(
        f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}."
      )
    self._prev_action[:] = self._action
    self._action[:] = action.to(self.device)
    # Split and apply.
    idx = 0
    for term in self._terms.values():
      term_actions = action[:, idx : idx + term.action_dim]
      term.process_actions(term_actions)
      idx += term.action_dim

  def apply_action(self) -> None:
    for term in self._terms.values():
      term.apply_actions()

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    idx = 0
    for name, term in self._terms.items():
      term_actions = self._action[env_idx, idx : idx + term.action_dim].cpu()
      terms.append((name, term_actions.tolist()))
      idx += term.action_dim
    return terms

  def _prepare_terms(self):
    self._term_names: list[str] = list()
    self._terms: dict[str, ActionTerm] = dict()

    for term_name, term_cfg in self.cfg.items():
      term_cfg: ActionTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      term = term_cfg.class_type(term_cfg, self._env)
      self._term_names.append(term_name)
      self._terms[term_name] = term
