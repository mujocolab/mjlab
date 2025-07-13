from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Sequence
from mjlab.managers.manager_base import ManagerBase, ManagerTermBase
from mjlab.managers.manager_term_config import RewardTermCfg

from mjlab.utils.dataclasses import get_terms

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv


class RewardManager(ManagerBase):
  _env: ManagerBasedRLEnv

  def __init__(self, cfg: object, env: ManagerBasedRLEnv):
    super().__init__(cfg=cfg, env=env)

    self._episode_sums = dict()
    for term_name in self._term_names:
      self._episode_sums[term_name] = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device
      )
    self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    self._step_reward = torch.zeros(
      (self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device
    )

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  # Methods.

  def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
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
    for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
      if term_cfg.weight == 0.0:
        self._step_rewardp[:, term_idx] = 0.0
        continue
      value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
      self._reward_buf += value
      self._episode_sums[name] += value
      self._step_reward[:, term_idx] = value / dt
    return self._reward_buf

  def get_active_iterable_terms(self, env_idx):
    terms = []
    for idx, name in enumerate(self._term_names):
      terms.append((name, [self._step_reward[env_idx, idx].cpu().item()]))
    return terms

  def _prepare_terms(self):
    self._term_names: list[str] = list()
    self._term_cfgs: list[RewardTermCfg] = list()
    self._class_term_cfgs: list[RewardTermCfg] = list()

    cfg_items = get_terms(self.cfg, RewardTermCfg).items()
    for term_name, term_cfg in cfg_items:
      term_cfg: RewardTermCfg
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if isinstance(term_cfg.func, ManagerTermBase):
        self._class_term_cfgs.append(term_cfg)
