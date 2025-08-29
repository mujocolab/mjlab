from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBase
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.utils.dataclasses import get_terms

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class CommandTerm(ManagerTermBase):
  """Base class for command terms."""

  def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self.cfg = cfg
    self.metrics = dict()
    self.time_left = torch.zeros(self.num_envs, device=self.device)
    self.command_counter = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )

  def debug_vis(self, scn):
    if self.cfg.debug_vis:
      self._debug_vis_impl(scn)

  def _debug_vis_impl(self, scn):
    pass

  @property
  @abc.abstractmethod
  def command(self):
    raise NotImplementedError

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    extras = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    self.command_counter[env_ids] = 0
    self._resample(env_ids)
    return extras

  def compute(self, dt: float) -> None:
    self._update_metrics()
    self.time_left -= dt
    resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
    if len(resample_env_ids) > 0:
      self._resample(resample_env_ids)
    self._update_command()

  def _resample(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) != 0:
      self.time_left[env_ids] = self.time_left[env_ids].uniform_(
        *self.cfg.resampling_time_range
      )
      self._resample_command(env_ids)
      self.command_counter[env_ids] += 1

  @abc.abstractmethod
  def _update_metrics(self) -> None:
    """Update the metrics based on the current state."""
    raise NotImplementedError

  @abc.abstractmethod
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample the command for the specified environments."""
    raise NotImplementedError

  @abc.abstractmethod
  def _update_command(self) -> None:
    """Update the command based on the current state."""
    raise NotImplementedError


class CommandManager(ManagerBase):
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: object, env: ManagerBasedRlEnv):
    self._terms: dict[str, CommandTerm] = dict()

    self.cfg = cfg
    super().__init__(env)
    self._commands = dict()

  def __str__(self) -> str:
    msg = f"<CommandManager> contains {len(self._terms.values())} active terms.\n"
    table = PrettyTable()
    table.title = "Active Command Terms"
    table.field_names = ["Index", "Name", "Type"]
    table.align["Name"] = "l"
    for index, (name, term) in enumerate(self._terms.items()):
      table.add_row([index, name, term.__class__.__name__])
    msg += table.get_string()
    msg += "\n"
    return msg

  def debug_vis(self, scn):
    for term in self._terms.values():
      term.debug_vis(scn)

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms.keys())

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    idx = 0
    for name, term in self._terms.items():
      terms.append((name, term.command[env_idx].cpu().tolist()))
      idx += term.command.shape[1]
    return terms

  def reset(self, env_ids: torch.Tensor | None) -> dict[str, torch.Tensor]:
    extras = {}
    for name, term in self._terms.items():
      metrics = term.reset(env_ids=env_ids)
      for metric_name, metric_value in metrics.items():
        extras[f"Metrics/{name}/{metric_name}"] = metric_value
    return extras

  def compute(self, dt: float):
    for term in self._terms.values():
      term.compute(dt)

  def get_command(self, name: str) -> torch.Tensor:
    return self._terms[name].command

  def get_term(self, name: str) -> CommandTerm:
    return self._terms[name]

  def _prepare_terms(self):
    cfg_items = get_terms(self.cfg, CommandTermCfg).items()
    for term_name, term_cfg in cfg_items:
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      term = term_cfg.class_type(term_cfg, self._env)
      if not isinstance(term, CommandTerm):
        raise TypeError(
          f"Returned object for the term {term_name} is not of type CommandType."
        )
      self._terms[term_name] = term
