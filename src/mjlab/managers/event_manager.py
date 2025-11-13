"""Event manager for orchestrating operations based on different simulation events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase
from mjlab.managers.manager_term_config import EventMode, EventTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class EventManager(ManagerBase):
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: dict[str, EventTermCfg], env: ManagerBasedRlEnv):
    self.cfg = cfg
    self._mode_term_names: dict[EventMode, list[str]] = dict()
    self._mode_term_cfgs: dict[EventMode, list[EventTermCfg]] = dict()
    self._mode_class_term_cfgs: dict[EventMode, list[EventTermCfg]] = dict()
    self._domain_randomization_fields: list[str] = list()

    super().__init__(env=env)

  def __str__(self) -> str:
    msg = f"<EventManager> contains {len(self._mode_term_names)} active terms.\n"
    for mode in self._mode_term_names:
      table = PrettyTable()
      table.title = f"Active Event Terms in Mode: '{mode}'"
      if mode == "interval":
        table.field_names = ["Index", "Name", "Interval time range (s)"]
        table.align["Name"] = "l"
        for index, (name, cfg) in enumerate(
          zip(self._mode_term_names[mode], self._mode_term_cfgs[mode], strict=False)
        ):
          table.add_row([index, name, cfg.interval_range_s])
      else:
        table.field_names = ["Index", "Name"]
        table.align["Name"] = "l"
        for index, name in enumerate(self._mode_term_names[mode]):
          table.add_row([index, name])
      msg += table.get_string()
      msg += "\n"
    if self._domain_randomization_fields:
      table = PrettyTable()
      table.title = "Domain Randomization Fields"
      table.field_names = ["Index", "Field Name"]
      table.align["Field Name"] = "l"
      for index, field in enumerate(self._domain_randomization_fields):
        table.add_row([index, field])
      msg += table.get_string()
      msg += "\n"
    return msg

  # Properties.

  @property
  def active_terms(self) -> dict[EventMode, list[str]]:
    return self._mode_term_names

  @property
  def available_modes(self) -> list[EventMode]:
    return list(self._mode_term_names.keys())

  @property
  def domain_randomization_fields(self) -> tuple[str, ...]:
    return tuple(self._domain_randomization_fields)

  # Methods.

  def get_term_cfg(self, term_name: str) -> EventTermCfg:
    """Get the configuration of a specific event term by name."""
    for mode in self._mode_term_names:
      if term_name in self._mode_term_names[mode]:
        index = self._mode_term_names[mode].index(term_name)
        return self._mode_term_cfgs[mode][index]
    raise ValueError(f"Event term '{term_name}' not found in active terms.")

  def reset(self, env_ids: torch.Tensor | None = None):
    for mode_cfg in self._mode_class_term_cfgs.values():
      for term_cfg in mode_cfg:
        term_cfg.func.reset(env_ids=env_ids)
    if env_ids is None:
      num_envs = self._env.num_envs
    else:
      num_envs = len(env_ids)
    if "interval" in self._mode_term_cfgs:
      for index, term_cfg in enumerate(self._mode_class_term_cfgs["interval"]):
        if not term_cfg.is_global_time:
          assert term_cfg.interval_range_s is not None
          lower, upper = term_cfg.interval_range_s
          sampled_interval = (
            torch.rand(num_envs, device=self.device) * (upper - lower) + lower
          )
          self._interval_term_time_left[index][env_ids] = sampled_interval
    return {}

  def apply(
    self,
    mode: EventMode,
    env_ids: torch.Tensor | slice | None = None,
    dt: float | None = None,
    global_env_step_count: int | None = None,
  ):
    if mode == "interval" and dt is None:
      raise ValueError(
        f"Event mode '{mode}' requires the time-step of the environment."
      )
    if mode == "interval" and env_ids is not None:
      raise ValueError(
        f"Event mode '{mode}' does not require environment indices. This is an undefined behavior"
        " as the environment indices are computed based on the time left for each environment."
      )
    if mode == "reset" and global_env_step_count is None:
      raise ValueError(
        f"Event mode '{mode}' requires the total number of environment steps to be provided."
      )

    for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
      if mode == "interval":
        time_left = self._interval_term_time_left[index]
        assert dt is not None
        time_left -= dt
        if term_cfg.is_global_time:
          if time_left < 1e-6:
            assert term_cfg.interval_range_s is not None
            lower, upper = term_cfg.interval_range_s
            sampled_interval = torch.rand(1) * (upper - lower) + lower
            self._interval_term_time_left[index][:] = sampled_interval
            term_cfg.func(self._env, None, **term_cfg.params)
        else:
          valid_env_ids = (time_left < 1e-6).nonzero().flatten()
          if len(valid_env_ids) > 0:
            assert term_cfg.interval_range_s is not None
            lower, upper = term_cfg.interval_range_s
            sampled_time = (
              torch.rand(len(valid_env_ids), device=self.device) * (upper - lower)
              + lower
            )
            self._interval_term_time_left[index][valid_env_ids] = sampled_time
            term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
      elif mode == "reset":
        assert global_env_step_count is not None
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
          if isinstance(env_ids, torch.Tensor):
            valid_env_ids = env_ids[valid_trigger]
          else:
            valid_env_ids = valid_trigger.nonzero().flatten()
          if len(valid_env_ids) > 0:
            self._reset_term_last_triggered_once[index][valid_env_ids] = True
            self._reset_term_last_triggered_step_id[index][valid_env_ids] = (
              global_env_step_count
            )
            term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
      else:
        term_cfg.func(self._env, env_ids, **term_cfg.params)

  def _prepare_terms(self) -> None:
    self._interval_term_time_left: list[torch.Tensor] = list()
    self._reset_term_last_triggered_step_id: list[torch.Tensor] = list()
    self._reset_term_last_triggered_once: list[torch.Tensor] = list()

    for term_name, term_cfg in self.cfg.items():
      term_cfg: EventTermCfg | None
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
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)
      if term_cfg.mode == "interval":
        if term_cfg.interval_range_s is None:
          raise ValueError(
            f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
          )
        if term_cfg.is_global_time:
          lower, upper = term_cfg.interval_range_s
          time_left = torch.rand(1) * (upper - lower) + lower
          self._interval_term_time_left.append(time_left)
        else:
          lower, upper = term_cfg.interval_range_s
          time_left = (
            torch.rand(self.num_envs, device=self.device) * (upper - lower) + lower
          )
          self._interval_term_time_left.append(time_left)
      elif term_cfg.mode == "reset":
        step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._reset_term_last_triggered_step_id.append(step_count)
        no_trigger = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._reset_term_last_triggered_once.append(no_trigger)

      if term_cfg.domain_randomization:
        field_name = term_cfg.params["field"]
        if field_name not in self._domain_randomization_fields:
          self._domain_randomization_fields.append(field_name)
