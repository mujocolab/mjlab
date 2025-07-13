from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from mjlab.managers.manager_base import ManagerBase
from mjlab.managers.manager_term_config import EventTermCfg

from mjlab.utils.dataclasses import get_terms

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv


class EventManager(ManagerBase):
  _env: ManagerBasedEnv

  def __init__(self, cfg: object, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

  # Properties.

  @property
  def active_terms(self) -> dict[str, list[str]]:
    return self._mode_term_names

  @property
  def available_modes(self) -> list[str]:
    return list(self._mode_term_names.keys())

  # Methods.

  def reset(self, env_ids=None):
    for mode_cfg in self._mode_class_term_cfgs.values():
      for term_cfg in mode_cfg:
        term_cfg.func.reset(env_ids=env_ids)
    return {}

  def apply(
    self,
    mode: str,
    env_ids: Sequence[int] | None = None,
    dt: float | None = None,
    global_env_step_count: int | None = None,
  ):
    if mode == "reset" and global_env_step_count is None:
      raise ValueError

    for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
      if mode == "interval":
        pass
      elif mode == "reset":
        min_step_count = term_cfg.min_step_count_between_reset
        if env_ids is None:
          env_ids = slice(None)
        if min_step_count == 0:
          self._reset_term_last_triggered_step_id[index][env_ids] = (
            global_env_step_count
          )
          self._reset_term_last_triggered_once[index][env_ids] = True
          term_cfg.func(self._env, env_ids, **term_cfg.params)
        else:
          last_triggered_step = self._reset_term_last_triggered_step_id[index][env_ids]
          triggered_at_least_once = self._reset_term_last_triggered_once[index][env_ids]
          steps_since_triggered = global_env_step_count - last_triggered_step
          valid_trigger = steps_since_triggered >= min_step_count
          valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once
          if env_ids == slice(None):
            valid_env_ids = valid_trigger.nonzero().flatten()
          else:
            valid_env_ids = env_ids[valid_trigger]
          if len(valid_env_ids) > 0:
            self._reset_term_last_triggered_once[index][valid_env_ids] = True
            self._reset_term_last_triggered_step_id[index][valid_env_ids] = (
              global_env_step_count
            )
            term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
      else:
        term_cfg.func(self._env, env_ids, **term_cfg.params)

  def _prepare_terms(self):
    self._mode_term_names: dict[str, list[str]] = dict()
    self._mode_term_cfgs: dict[str, list[EventTermCfg]] = dict()
    self._mode_class_term_cfgs: dict[str, list[EventTermCfg]] = dict()
    self._interval_term_time_left: list[torch.Tensor] = list()
    self._reset_term_last_triggered_step_id: list[torch.Tensor] = list()
    self._reset_term_last_triggered_once: list[torch.Tensor] = list()

    cfg_items = get_terms(self.cfg, EventTermCfg).items()
    for term_name, term_cfg in cfg_items:
      term_cfg: EventTermCfg
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      if term_cfg.mode not in self._mode_term_names:
        self._mode_term_names[term_cfg.mode] = list()
        self._mode_term_cfgs[term_cfg.mode] = list()
        self._mode_class_term_cfgs[term_cfg.mode] = list()
      self._mode_term_names[term_cfg.mode].append(term_name)
      self._mode_term_cfgs[term_cfg.mode].append(term_cfg)
      if term_cfg.mode == "interval":
        pass
      elif term_cfg.mode == "reset":
        step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._reset_term_last_triggered_step_id.append(step_count)
        no_trigger = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._reset_term_last_triggered_once.append(no_trigger)
