from __future__ import annotations
import abc

from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.managers.manager_base import ManagerBase, ManagerTermBase
from mjlab.utils.dataclasses import get_terms
from mjlab.managers.manager_term_config import ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv


class ActionTerm(ManagerTermBase):
  """Base class for action terms."""

  def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
    self.cfg = cfg
    self._env = env

  @property
  @abc.abstractmethod
  def action_dim(self) -> int:
    raise NotImplementedError

  @abc.abstractmethod
  def process_actions(self, actions: torch.Tensor):
    raise NotImplementedError

  @abc.abstractmethod
  def apply_actions(self):
    raise NotImplementedError


class ActionManager(ManagerBase):
  def __init__(self, cfg: object, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    # Create buffers to store actions.
    self._action = torch.zeros(
      (self.num_envs, self.total_action_dim), device=self.device
    )
    self._prev_action = torch.zeros_like(self._action)

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

  def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
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
    # Check if action dimension is valid.
    # Store the input actions.
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

    cfg_items = get_terms(self.cfg, ActionTermCfg).items()
    for term_name, term_cfg in cfg_items:
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      term = term_cfg.class_type(term_cfg, self._env)
      self._term_names.append(term_name)
      self._terms[term_name] = term
