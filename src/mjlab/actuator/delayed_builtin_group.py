"""Fused delayed builtin actuator group for batch buffer operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.builtin_group import _TARGET_TENSOR_MAP, BuiltinActuatorType
from mjlab.actuator.delayed_actuator import DelayedActuator, DelayedActuatorCfg
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.actuator.actuator import Actuator
  from mjlab.entity.data import EntityData


@dataclass
class _FusedGroup:
  """A single fused group of delayed builtin actuators sharing delay config."""

  target_attr: str
  target_ids: torch.Tensor
  ctrl_ids: torch.Tensor
  cfg: DelayedActuatorCfg
  absorbed_actuators: list[DelayedActuator] = field(default_factory=list)
  delay_buffer: DelayBuffer | None = field(default=None, init=False)


@dataclass
class DelayedBuiltinActuatorGroup:
  """Fuses delayed builtin actuators into a single buffer operation.

  When multiple ``DelayedActuator(BuiltinXxxActuator)`` instances share the same delay
  configuration and builtin type, their per-actuator buffer operations can be merged
  into one append + one compute per physics step instead of N, significantly reducing
  overhead.
  """

  _groups: list[_FusedGroup]

  @staticmethod
  def process(
    actuators: tuple[Actuator, ...] | list[Actuator],
  ) -> tuple[DelayedBuiltinActuatorGroup, tuple[Actuator, ...]]:
    """Extract delayed builtins and group by (type, transmission, delay cfg).

    Args:
      actuators: Custom actuators (already filtered by BuiltinActuatorGroup).

    Returns:
      A tuple of (fused group, remaining custom actuators).
    """
    grouped: dict[tuple, list[DelayedActuator]] = {}
    remaining: list[Actuator] = []

    for act in actuators:
      if not (
        isinstance(act, DelayedActuator)
        and isinstance(act.base_actuator, BuiltinActuatorType)
      ):
        remaining.append(act)
        continue

      cfg = act.cfg

      # Only fuse single-target delays. Multi-target on builtins is not meaningful
      # since each builtin type uses exactly one target channel.
      if isinstance(cfg.delay_target, tuple) and len(cfg.delay_target) > 1:
        remaining.append(act)
        continue

      delay_config_key = (
        cfg.delay_target
        if isinstance(cfg.delay_target, tuple)
        else (cfg.delay_target,),
        cfg.delay_min_lag,
        cfg.delay_max_lag,
        cfg.delay_hold_prob,
        cfg.delay_update_period,
        cfg.delay_per_env_phase,
      )
      key = (type(act.base_actuator), cfg.transmission_type, delay_config_key)
      grouped.setdefault(key, []).append(act)

    if not grouped:
      return DelayedBuiltinActuatorGroup([]), tuple(remaining)

    groups: list[_FusedGroup] = []
    for (base_type, transmission_type, _delay_cfg), acts in grouped.items():
      target_attr = _TARGET_TENSOR_MAP[(base_type, transmission_type)]
      target_ids = torch.cat([a.target_ids for a in acts], dim=0)
      ctrl_ids = torch.cat([a.ctrl_ids for a in acts], dim=0)
      groups.append(
        _FusedGroup(
          target_attr=target_attr,
          target_ids=target_ids,
          ctrl_ids=ctrl_ids,
          cfg=acts[0].cfg,
          absorbed_actuators=list(acts),
        )
      )

    return DelayedBuiltinActuatorGroup(groups), tuple(remaining)

  def initialize(self, num_envs: int, device: str) -> None:
    """Create delay buffers for each fused group."""
    for group in self._groups:
      cfg = group.cfg
      group.delay_buffer = DelayBuffer(
        min_lag=cfg.delay_min_lag,
        max_lag=cfg.delay_max_lag,
        batch_size=num_envs,
        device=device,
        hold_prob=cfg.delay_hold_prob,
        update_period=cfg.delay_update_period,
        per_env_phase=cfg.delay_per_env_phase,
      )
      # Alias the fused buffer into each absorbed actuator so that
      # per-actuator reset() and set_lags() operate on the shared buffer.
      target = (
        cfg.delay_target if isinstance(cfg.delay_target, str) else cfg.delay_target[0]
      )
      for act in group.absorbed_actuators:
        act._delay_buffers = {target: group.delay_buffer}

  def apply_controls(self, data: EntityData) -> None:
    """Apply delayed controls for all fused groups."""
    for group in self._groups:
      assert group.delay_buffer is not None
      target_tensor = getattr(data, group.target_attr)
      targets = target_tensor[:, group.target_ids]
      group.delay_buffer.append(targets)
      data.write_ctrl(group.delay_buffer.compute(), group.ctrl_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset all delay buffers for given env_ids."""
    for group in self._groups:
      if group.delay_buffer is not None:
        group.delay_buffer.reset(env_ids)

  def set_lags(
    self,
    lags: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """Set lags on all delay buffers."""
    for group in self._groups:
      if group.delay_buffer is not None:
        group.delay_buffer.set_lags(lags, env_ids)
