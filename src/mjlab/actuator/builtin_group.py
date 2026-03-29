"""Batch ctrl writes for builtin actuators.

Builtin actuators have identity compute(): they just pass the target through to ctrl.
Instead of looping over each actuator individually, this module reads the target tensor
and writes ctrl in one batched operation per actuator type. Delayed builtins go through
a shared DelayBuffer first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuator,
  BuiltinMuscleActuator,
  BuiltinPositionActuator,
  BuiltinVelocityActuator,
)
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.actuator.actuator import Actuator
  from mjlab.entity.data import EntityData

BuiltinActuatorType = (
  BuiltinMotorActuator
  | BuiltinMuscleActuator
  | BuiltinPositionActuator
  | BuiltinVelocityActuator
)

# Maps (actuator_type, transmission_type) to EntityData target tensor attribute name.
_TARGET_TENSOR_MAP: dict[tuple[type[BuiltinActuatorType], TransmissionType], str] = {
  (BuiltinPositionActuator, TransmissionType.JOINT): "joint_pos_target",
  (BuiltinVelocityActuator, TransmissionType.JOINT): "joint_vel_target",
  (BuiltinMotorActuator, TransmissionType.JOINT): "joint_effort_target",
  (BuiltinPositionActuator, TransmissionType.TENDON): "tendon_len_target",
  (BuiltinVelocityActuator, TransmissionType.TENDON): "tendon_vel_target",
  (BuiltinMotorActuator, TransmissionType.TENDON): "tendon_effort_target",
  (BuiltinMotorActuator, TransmissionType.SITE): "site_effort_target",
  (BuiltinMuscleActuator, TransmissionType.JOINT): "joint_effort_target",
  (BuiltinMuscleActuator, TransmissionType.TENDON): "tendon_effort_target",
}


@dataclass
class _FusedDelayGroup:
  """A fused group of delayed builtin actuators sharing delay config."""

  target_attr: str
  target_ids: torch.Tensor
  ctrl_ids: torch.Tensor
  min_lag: int
  max_lag: int
  hold_prob: float
  update_period: int
  per_env_phase: bool
  absorbed_actuators: list[Actuator] = field(default_factory=list)
  delay_buffer: DelayBuffer | None = field(default=None, init=False)


@dataclass
class BuiltinActuatorGroup:
  """Groups builtin actuators for batch processing."""

  _index_groups: dict[
    tuple[type[BuiltinActuatorType], TransmissionType],
    tuple[torch.Tensor, torch.Tensor],
  ]
  _delayed_groups: list[_FusedDelayGroup]

  @staticmethod
  def process(
    actuators: list[Actuator],
  ) -> tuple[BuiltinActuatorGroup, tuple[Actuator, ...]]:
    """Classify actuators into builtin groups and custom actuators.

    Non-delayed builtins go into direct index groups. Delayed builtins with matching
    delay config are fused into shared delay buffer groups. Everything else is returned
    as custom actuators.

    Args:
      actuators: List of initialized actuators to process.

    Returns:
      A tuple of (builtin group, custom actuators).
    """
    builtin_groups: dict[
      tuple[type[BuiltinActuatorType], TransmissionType], list[Actuator]
    ] = {}
    delayed_grouped: dict[tuple, list[Actuator]] = {}
    custom_actuators: list[Actuator] = []

    for act in actuators:
      if not isinstance(act, BuiltinActuatorType):
        custom_actuators.append(act)
        continue

      if not act.has_delay:
        key: tuple[type[BuiltinActuatorType], TransmissionType] = (
          type(act),
          act.cfg.transmission_type,
        )
        builtin_groups.setdefault(key, []).append(act)
        continue

      # Delayed builtin: fuse into shared delay buffer group.
      delay_key = (
        type(act),
        act.cfg.transmission_type,
        act.cfg.delay_min_lag,
        act.cfg.delay_max_lag,
        act.cfg.delay_hold_prob,
        act.cfg.delay_update_period,
        act.cfg.delay_per_env_phase,
      )
      delayed_grouped.setdefault(delay_key, []).append(act)

    # Build non-delayed index groups.
    index_groups: dict[
      tuple[type[BuiltinActuatorType], TransmissionType],
      tuple[torch.Tensor, torch.Tensor],
    ] = {
      key: (
        torch.cat([a.target_ids for a in acts], dim=0),
        torch.cat([a.ctrl_ids for a in acts], dim=0),
      )
      for key, acts in builtin_groups.items()
    }

    # Build delayed fused groups.
    delayed_groups: list[_FusedDelayGroup] = []
    for (base_type, transmission_type, *_), acts in delayed_grouped.items():
      target_attr = _TARGET_TENSOR_MAP[(base_type, transmission_type)]
      target_ids = torch.cat([a.target_ids for a in acts], dim=0)
      ctrl_ids = torch.cat([a.ctrl_ids for a in acts], dim=0)
      cfg = acts[0].cfg
      delayed_groups.append(
        _FusedDelayGroup(
          target_attr=target_attr,
          target_ids=target_ids,
          ctrl_ids=ctrl_ids,
          min_lag=cfg.delay_min_lag,
          max_lag=cfg.delay_max_lag,
          hold_prob=cfg.delay_hold_prob,
          update_period=cfg.delay_update_period,
          per_env_phase=cfg.delay_per_env_phase,
          absorbed_actuators=list(acts),
        )
      )

    return BuiltinActuatorGroup(index_groups, delayed_groups), tuple(custom_actuators)

  def initialize(self, num_envs: int, device: str) -> None:
    """Create fused delay buffers for delayed builtin groups."""
    for group in self._delayed_groups:
      group.delay_buffer = DelayBuffer(
        min_lag=group.min_lag,
        max_lag=group.max_lag,
        batch_size=num_envs,
        device=device,
        hold_prob=group.hold_prob,
        update_period=group.update_period,
        per_env_phase=group.per_env_phase,
      )
      # Alias the fused buffer into each absorbed actuator so that per-actuator reset
      # and set_lags operate on the shared buffer.
      for act in group.absorbed_actuators:
        act._delay_buffer = group.delay_buffer

  def apply_controls(self, data: EntityData) -> None:
    """Write builtin actuator controls to simulation data."""
    # Non-delayed: direct write.
    for (actuator_type, transmission_type), (
      target_ids,
      ctrl_ids,
    ) in self._index_groups.items():
      attr_name = _TARGET_TENSOR_MAP[(actuator_type, transmission_type)]
      target_tensor = getattr(data, attr_name)
      data.write_ctrl(target_tensor[:, target_ids], ctrl_ids)

    # Delayed: append to buffer, compute delayed value, write.
    for group in self._delayed_groups:
      assert group.delay_buffer is not None
      target_tensor = getattr(data, group.target_attr)
      targets = target_tensor[:, group.target_ids]
      group.delay_buffer.append(targets)
      data.write_ctrl(group.delay_buffer.compute(), group.ctrl_ids)
