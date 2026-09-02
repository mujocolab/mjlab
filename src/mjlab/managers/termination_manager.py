"""Termination manager for computing done signals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from mjlab.utils.torch import as_index

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
  """Configuration for a termination term."""

  time_out: bool = False
  """Whether the term contributes towards episodic timeouts."""


class TerminationManager(ManagerBase):
  """Manages termination conditions for the environment.

  The termination manager aggregates multiple termination terms to compute
  episode done signals. Terms can be either truncations (time-based) or
  terminations (failure conditions).
  """

  _env: ManagerBasedRlEnv

  def __init__(self, cfg: dict[str, TerminationTermCfg], env: ManagerBasedRlEnv):
    self._term_names: list[str] = list()
    self._term_cfgs: list[TerminationTermCfg] = list()
    self._class_term_cfgs: list[TerminationTermCfg] = list()

    self.cfg = deepcopy(cfg)
    super().__init__(env)

    # All per-term done flags live in one (num_envs, num_terms) buffer so the
    # truncated/terminated reductions and the reset-time episode counts are each a
    # single batched op. ``_term_dones`` exposes the per-term columns as views.
    self._dones_buf = torch.zeros(
      (self.num_envs, len(self._term_names)), device=self.device, dtype=torch.bool
    )
    self._term_dones = {
      term_name: self._dones_buf[:, idx]
      for idx, term_name in enumerate(self._term_names)
    }
    self._time_out_sel = self._column_index(
      [idx for idx, cfg in enumerate(self._term_cfgs) if cfg.time_out]
    )
    self._terminated_sel = self._column_index(
      [idx for idx, cfg in enumerate(self._term_cfgs) if not cfg.time_out]
    )
    self._truncated_buf = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._terminated_buf = torch.zeros_like(self._truncated_buf)

  def __str__(self) -> str:
    msg = f"<TerminationManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Termination Terms"
    table.field_names = ["Index", "Name", "Time Out"]
    table.align["Name"] = "l"
    for index, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      table.add_row([index, name, term_cfg.time_out])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  @property
  def dones(self) -> torch.Tensor:
    return self._truncated_buf | self._terminated_buf

  @property
  def time_outs(self) -> torch.Tensor:
    return self._truncated_buf

  @property
  def terminated(self) -> torch.Tensor:
    return self._terminated_buf

  # Methods.

  def reset(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    if env_ids is None:
      env_ids = slice(None)
    # One batched reduction over the selected envs. The counts are logged as 0-d
    # tensors: calling .item() per term would synchronize with the device on
    # every reset.
    counts = self._dones_buf[env_ids].sum(dim=0)
    extras = {
      "Episode_Termination/" + key: counts[idx]
      for idx, key in enumerate(self._term_names)
    }
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return extras

  def compute(self) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=False):
      value = term_cfg.func(self._env, **term_cfg.params)
      self._check_term_shape(name, value)
      if value.dtype != torch.bool:
        value = value.bool()
      values.append(value)
    if values:
      # One launch gathers every term into the (num_envs, num_terms) buffer.
      torch.stack(values, dim=1, out=self._dones_buf)
    self._reduce_columns(self._time_out_sel, self._truncated_buf)
    self._reduce_columns(self._terminated_sel, self._terminated_buf)
    return self._truncated_buf | self._terminated_buf

  def _column_index(self, columns: list[int]) -> slice | torch.Tensor | None:
    """Index selecting the given columns of ``_dones_buf`` (None if empty)."""
    if not columns:
      return None
    return as_index(torch.tensor(columns, dtype=torch.long, device=self.device))

  def _reduce_columns(
    self, sel: slice | torch.Tensor | None, out: torch.Tensor
  ) -> None:
    """OR the selected done columns into ``out``."""
    if sel is None:
      out.fill_(False)
    else:
      torch.any(self._dones_buf[:, sel], dim=1, out=out)

  def get_term(self, name: str) -> torch.Tensor:
    return self._term_dones[name]

  def get_term_cfg(self, term_name: str) -> TerminationTermCfg:
    if term_name not in self._term_names:
      raise ValueError(f"Term '{term_name}' not found in active terms.")
    return self._term_cfgs[self._term_names.index(term_name)]

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    for key in self._term_dones.keys():
      terms.append((key, [self._term_dones[key][env_idx].float().cpu().item()]))
    return terms

  def _prepare_terms(self):
    for term_name, term_cfg in self.cfg.items():
      term_cfg: TerminationTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._class_term_cfgs.append(term_cfg)
