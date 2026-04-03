"""Metrics manager for logging custom per-step metrics during training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MetricsTermCfg(ManagerTermBaseCfg):
  """Configuration for a metrics term.

  Attributes:
    per_substep: If True, evaluate this term once per physics substep inside the
    decimation loop and report the per-step mean. Only the integrated state
    (qpos, qvel, act) is current mid-loop; all derived quantities (xpos, xquat,
    site_xpos, actuator_force, contacts, ...) are stale.
    reduce: How to aggregate per-step values into an episode metric.
    ``"mean"`` (default) reports ``sum / step_count``. ``"last"`` reports
    the value from the final step of the episode, which is useful for binary
    success metrics that should not be averaged over timesteps.
  """

  per_substep: bool = False
  reduce: Literal["mean", "last"] = "mean"


class MetricsManager(ManagerBase):
  """Accumulates per-step metric values, reports episode averages.

  Unlike rewards, metrics have no weight, no dt scaling, and no
  normalization by episode length. Episode values are true per-step
  averages (sum / step_count), so a metric in [0,1] stays in [0,1]
  in the logger.
  """

  _env: ManagerBasedRlEnv

  def __init__(self, cfg: dict[str, MetricsTermCfg], env: ManagerBasedRlEnv):
    self._term_names: list[str] = list()
    self._term_cfgs: list[MetricsTermCfg] = list()
    self._class_term_cfgs: list[MetricsTermCfg] = list()
    self._step_term_indices: list[int] = list()
    self._substep_term_indices: list[int] = list()

    self.cfg = deepcopy(cfg)
    super().__init__(env=env)

    self._episode_sums: dict[str, torch.Tensor] = {}
    for term_name in self._term_names:
      self._episode_sums[term_name] = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device
      )
    # Pre-resolved tensor refs for substep terms to avoid dict lookups in
    # the hot loop.
    self._substep_accum: list[torch.Tensor] = []
    self._substep_episode_sums: list[torch.Tensor] = []
    for idx in self._substep_term_indices:
      name = self._term_names[idx]
      buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
      self._substep_accum.append(buf)
      self._substep_episode_sums.append(self._episode_sums[name])
    self._substep_count: int = 0
    self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._step_values = torch.zeros(
      (self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device
    )

  def __str__(self) -> str:
    msg = f"<MetricsManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Metrics Terms"
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

  def reset(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    if env_ids is None:
      env_ids = slice(None)
    extras = {}
    counts = self._step_count[env_ids].float()
    # Avoid division by zero for envs that haven't stepped.
    safe_counts = torch.clamp(counts, min=1.0)
    for idx, key in enumerate(self._episode_sums):
      if self._term_cfgs[idx].reduce == "last":
        extras["Episode_Metrics/" + key] = torch.mean(self._step_values[env_ids, idx])
      else:
        extras["Episode_Metrics/" + key] = torch.mean(
          self._episode_sums[key][env_ids] / safe_counts
        )
      self._episode_sums[key][env_ids] = 0.0
    self._step_count[env_ids] = 0
    for buf in self._substep_accum:
      buf[env_ids] = 0.0
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return extras

  def compute_substep(self) -> None:
    """Accumulate per-substep metric values inside the decimation loop.

    No-op when no ``per_substep`` terms are configured.
    """
    if not self._substep_term_indices:
      return
    for i, idx in enumerate(self._substep_term_indices):
      value = self._term_cfgs[idx].func(self._env, **self._term_cfgs[idx].params)
      self._substep_accum[i] += value
    self._substep_count += 1

  def compute(self) -> None:
    self._step_count += 1
    if self._substep_term_indices and self._substep_count > 0:
      for i, idx in enumerate(self._substep_term_indices):
        avg = self._substep_accum[i] / self._substep_count
        self._substep_episode_sums[i] += avg
        self._step_values[:, idx] = avg
        self._substep_accum[i].zero_()
      self._substep_count = 0
    for idx in self._step_term_indices:
      name = self._term_names[idx]
      term_cfg = self._term_cfgs[idx]
      value = term_cfg.func(self._env, **term_cfg.params)
      self._episode_sums[name] += value
      self._step_values[:, idx] = value

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    for idx, name in enumerate(self._term_names):
      terms.append((name, [self._step_values[env_idx, idx].cpu().item()]))
    return terms

  def _prepare_terms(self):
    for term_name, term_cfg in self.cfg.items():
      term_cfg: MetricsTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      idx = len(self._term_names)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if term_cfg.per_substep:
        self._substep_term_indices.append(idx)
      else:
        self._step_term_indices.append(idx)
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._class_term_cfgs.append(term_cfg)


class NullMetricsManager:
  """Placeholder for absent metrics manager that safely no-ops all operations."""

  def __init__(self):
    self.active_terms: list[str] = []
    self.cfg = None

  def __str__(self) -> str:
    return "<NullMetricsManager> (inactive)"

  def __repr__(self) -> str:
    return "NullMetricsManager()"

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    return []

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
    return {}

  def compute_substep(self) -> None:
    pass

  def compute(self) -> None:
    pass
