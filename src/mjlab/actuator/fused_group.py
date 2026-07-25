"""Batch compute for stateless-control-law actuators.

Actuators like IdealPdActuator and DcMotorActuator evaluate their control law in
PyTorch every physics substep. Done per actuator, a robot split into several
groups (e.g. one per motor type) issues that many gathers, delay-buffer updates,
control-law evaluations, and ctrl writes each substep. At small env counts (play
mode with a synchronizing viewer) this is host-launch-bound and noticeably slow,
whereas built-in actuators are already fused into one batched write.

This module brings the same fusion to actuators whose control output is a
stateless function of per-target parameters and the command. Such actuators apply
their control law through the shared compute inherited from IdealPdActuator;
keeping that compute is exactly what marks an actuator as fusable. Actuators with
a custom compute (built-ins, XML, learned networks) are skipped, with no opt-out
flag required.

Fusable actuators sharing an exact type, transmission type, and delay config are
fused into one group: their per-target parameters and indices are concatenated so
the gather, optional delay, control law, and ctrl write each happen once for the
whole group, behind a single shared DelayBuffer. SITE transmission is not fused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import ActuatorCmd, TransmissionType, delay_command
from mjlab.actuator.pd_actuator import IdealPdActuator
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.actuator.actuator import Actuator
  from mjlab.entity.data import EntityData

# Per transmission type, the EntityData attribute names for the five fields the
# control law reads: (position_target, velocity_target, effort_target, position,
# velocity).
_FIELD_MAP: dict[TransmissionType, tuple[str, str, str, str, str]] = {
  TransmissionType.JOINT: (
    "joint_pos_target",
    "joint_vel_target",
    "joint_effort_target",
    "joint_pos",
    "joint_vel",
  ),
  TransmissionType.TENDON: (
    "tendon_len_target",
    "tendon_vel_target",
    "tendon_effort_target",
    "tendon_len",
    "tendon_vel",
  ),
}


def _is_fusable(act: Actuator) -> bool:
  """An actuator is fusable iff it keeps the shared stateless-law compute.

  IdealPd and DcMotor apply their control law through that inherited compute;
  anything with a custom compute (built-ins, XML, and learned actuators,
  including DcMotor subclasses that override it) is not fusable. Comparing the
  bound function is immune to the inheritance trap an isinstance check would hit.
  """
  return (
    type(act).compute is IdealPdActuator.compute
    and act.cfg.transmission_type in _FIELD_MAP
  )


def _require(t: torch.Tensor | None) -> torch.Tensor:
  assert t is not None
  return t


@dataclass
class _FusedGroup:
  """A fused group of actuators sharing type, transmission, and delay config."""

  actuator_type: type[Actuator]
  transmission_type: TransmissionType
  target_ids: torch.Tensor
  ctrl_ids: torch.Tensor
  min_lag: int
  max_lag: int
  hold_prob: float
  update_period: int
  per_env_phase: bool
  absorbed_actuators: list[Actuator] = field(default_factory=list)
  params: dict[str, torch.Tensor] = field(default_factory=dict, init=False)
  delay_buffer: DelayBuffer | None = field(default=None, init=False)

  @property
  def has_delay(self) -> bool:
    return self.max_lag > 0


@dataclass
class FusedActuatorGroup:
  """Groups stateless-control-law actuators for batch processing."""

  _groups: list[_FusedGroup]

  @staticmethod
  def process(
    actuators: tuple[Actuator, ...] | list[Actuator],
  ) -> tuple[FusedActuatorGroup, tuple[Actuator, ...]]:
    """Classify actuators into fused groups and remaining custom actuators.

    Fusable actuators (see _is_fusable) with JOINT or TENDON transmission are
    fused by (exact type, transmission type, delay config). Everything else is
    returned unchanged for the per-actuator path.

    Args:
      actuators: List of initialized actuators to process.

    Returns:
      A tuple of (fused group, remaining custom actuators).
    """
    grouped: dict[tuple, list[Actuator]] = {}
    remaining: list[Actuator] = []

    for act in actuators:
      if not _is_fusable(act):
        remaining.append(act)
        continue
      key = (
        type(act),
        act.cfg.transmission_type,
        act.cfg.delay_min_lag,
        act.cfg.delay_max_lag,
        act.cfg.delay_hold_prob,
        act.cfg.delay_update_period,
        act.cfg.delay_per_env_phase,
      )
      grouped.setdefault(key, []).append(act)

    groups: list[_FusedGroup] = []
    for (actuator_type, transmission_type, *_), acts in grouped.items():
      cfg = acts[0].cfg
      groups.append(
        _FusedGroup(
          actuator_type=actuator_type,
          transmission_type=transmission_type,
          target_ids=torch.cat([a.target_ids for a in acts], dim=0),
          ctrl_ids=torch.cat([a.ctrl_ids for a in acts], dim=0),
          min_lag=cfg.delay_min_lag,
          max_lag=cfg.delay_max_lag,
          hold_prob=cfg.delay_hold_prob,
          update_period=cfg.delay_update_period,
          per_env_phase=cfg.delay_per_env_phase,
          absorbed_actuators=list(acts),
        )
      )

    return FusedActuatorGroup(groups), tuple(remaining)

  def initialize(self, num_envs: int, device: str) -> None:
    """Concatenate parameters and create shared delay buffers for fused groups."""
    for group in self._groups:
      # Concatenate each control-law parameter across absorbed actuators, then
      # alias every actuator's parameter back to a slice view of the fused tensor.
      #
      # Invariant: after this an absorbed actuator's parameter tensors are views
      # into the group tensors. Mutations must be in-place (e.g. set_gains and
      # set_effort_limit assign into a slice) so they write through to what the
      # group computes with. Rebinding the attribute would silently detach it; the
      # only rebind site is the actuator's initialize(), which runs before fusion.
      for name in group.actuator_type.param_names:
        group.params[name] = torch.cat(
          [_require(getattr(a, name)) for a in group.absorbed_actuators], dim=1
        )
      offset = 0
      for act in group.absorbed_actuators:
        n = act.target_ids.numel()
        for name in group.actuator_type.param_names:
          setattr(act, name, group.params[name][:, offset : offset + n])
        offset += n

      if group.has_delay:
        group.delay_buffer = DelayBuffer(
          min_lag=group.min_lag,
          max_lag=group.max_lag,
          batch_size=num_envs,
          device=device,
          hold_prob=group.hold_prob,
          update_period=group.update_period,
          per_env_phase=group.per_env_phase,
        )
        # Alias the shared buffer into each absorbed actuator so per-actuator
        # reset and set_lags operate on it.
        for act in group.absorbed_actuators:
          act._delay_buffer = group.delay_buffer

  def apply_controls(self, data: EntityData) -> None:
    """Compute and write fused actuator controls to simulation data."""
    for group in self._groups:
      pos_attr, vel_attr, eff_attr, cur_pos_attr, cur_vel_attr = _FIELD_MAP[
        group.transmission_type
      ]
      ids = group.target_ids
      pos_target = getattr(data, pos_attr)[:, ids]
      vel_target = getattr(data, vel_attr)[:, ids]
      effort_target = getattr(data, eff_attr)[:, ids]
      pos = getattr(data, cur_pos_attr)[:, ids]
      vel = getattr(data, cur_vel_attr)[:, ids]

      if group.delay_buffer is not None:
        pos_target, vel_target, effort_target = delay_command(
          group.delay_buffer, pos_target, vel_target, effort_target
        )

      cmd = ActuatorCmd(
        position_target=pos_target,
        velocity_target=vel_target,
        effort_target=effort_target,
        pos=pos,
        vel=vel,
      )
      torques = group.actuator_type.control_law(group.params, cmd)
      data.write_ctrl(torques, group.ctrl_ids)
