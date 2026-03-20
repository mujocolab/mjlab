"""Reward manager for computing reward signals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class RewardTermCfg(ManagerTermBaseCfg):
  """Configuration for a reward term."""

  func: Any
  """The callable that computes this reward term's value."""

  weight: float
  """Weight multiplier for this reward term."""


class RewardManager(ManagerBase):
  """Manages reward computation by aggregating weighted reward terms.

  Reward Scaling Behavior:
    By default, rewards are scaled by the environment step duration (dt). This
    normalizes cumulative episodic rewards across different simulation frequencies.
    The scaling can be disabled via the ``scale_by_dt`` parameter.

    When ``scale_by_dt=True`` (default):
      - ``reward_buf`` (returned by ``compute()``) = raw_value * weight * dt
      - ``_episode_sums`` (cumulative rewards) are scaled by dt
      - ``Episode_Reward/*`` logged metrics are scaled by dt

    When ``scale_by_dt=False``:
      - ``reward_buf`` = raw_value * weight (no dt scaling)

    Regardless of the scaling setting:
      - ``_step_reward`` (via ``get_active_iterable_terms()``) always contains
        the unscaled reward rate (raw_value * weight)
  """

  _env: ManagerBasedRlEnv

  def __init__(
    self,
    cfg: dict[str, RewardTermCfg],
    env: ManagerBasedRlEnv,
    *,
    scale_by_dt: bool = True,
  ):
    self._term_names: list[str] = list()
    self._term_cfgs: list[RewardTermCfg] = list()
    self._class_term_cfgs: list[RewardTermCfg] = list()
    self._scale_by_dt = scale_by_dt

    self.cfg = deepcopy(cfg)
    super().__init__(env=env)
    self._episode_sums = dict()
    for term_name in self._term_names:
      self._episode_sums[term_name] = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device
      )
    self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    self._step_reward = torch.zeros(
      (self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device
    )

  def __str__(self) -> str:
    msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Reward Terms"
    table.field_names = ["Index", "Name", "Weight"]
    table.align["Name"] = "l"
    table.align["Weight"] = "r"
    for index, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      table.add_row([index, name, term_cfg.weight])
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
    for key in self._episode_sums.keys():
      episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
      extras["Episode_Reward/" + key] = (
        episodic_sum_avg / self._env.max_episode_length_s
      )
      self._episode_sums[key][env_ids] = 0.0
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return extras

  def compute(self, dt: float) -> torch.Tensor:
    self._reward_buf[:] = 0.0
    scale = dt if self._scale_by_dt else 1.0
    for term_idx, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      if term_cfg.weight == 0.0:
        self._step_reward[:, term_idx] = 0.0
        continue
      value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale
      # NaN/Inf can occur from corrupted physics state; zero them to avoid policy crash.
      value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
      self._reward_buf += value
      self._episode_sums[name] += value
      self._step_reward[:, term_idx] = value / scale
    return self._reward_buf

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Delegate debug visualization to class-based reward terms."""
    for _, func in self.get_visualizable_terms():
      func.debug_vis(visualizer)

  def get_visualizable_terms(self) -> list[tuple[str, Any]]:
    """Return ``(name, func)`` pairs for class-based terms with debug_vis."""
    results: list[tuple[str, Any]] = []
    for term_cfg in self._class_term_cfgs:
      if not hasattr(term_cfg.func, "debug_vis"):
        continue
      name = next(
        n
        for n, c in zip(self._term_names, self._term_cfgs, strict=False)
        if c is term_cfg
      )
      results.append((name, term_cfg.func))
    return results

  def get_active_iterable_terms(self, env_idx):
    terms = []
    for idx, name in enumerate(self._term_names):
      terms.append((name, [self._step_reward[env_idx, idx].cpu().item()]))
    return terms

  def get_term_cfg(self, term_name: str) -> RewardTermCfg:
    if term_name not in self._term_names:
      raise ValueError(f"Term '{term_name}' not found in active terms.")
    return self._term_cfgs[self._term_names.index(term_name)]

  def _prepare_terms(self):
    for term_name, term_cfg in self.cfg.items():
      term_cfg: RewardTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._class_term_cfgs.append(term_cfg)
