"""Generic delayed actuator wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class DelayedActuatorCfg(ActuatorCfg):
  """Configuration for delayed actuator wrapper.

  Wraps any actuator config to add delay functionality. Delays are quantized
  to physics timesteps (not control timesteps). For example, with 500Hz physics
  and 50Hz control (decimation=10), a lag of 2 represents a 4ms delay (2 physics
  steps).
  """

  base_cfg: ActuatorCfg
  """Configuration for the underlying actuator."""

  delay_target: (
    Literal["position", "velocity", "effort"]
    | tuple[Literal["position", "velocity", "effort"], ...]
  ) = "position"
  """Which command target(s) to delay.

  Can be a single string like 'position', or a tuple of strings like
  ('position', 'velocity', 'effort') to delay multiple targets together.
  """

  delay_min_lag: int = 0
  """Minimum delay lag in physics timesteps."""

  delay_max_lag: int = 0
  """Maximum delay lag in physics timesteps."""

  delay_hold_prob: float = 0.0
  """Probability of keeping previous lag when updating."""

  delay_update_period: int = 0
  """Period for updating delays in physics timesteps (0 = every step)."""

  delay_per_env_phase: bool = True
  """Whether each environment has a different phase offset."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> DelayedActuator:
    base_actuator = self.base_cfg.build(entity, joint_ids, joint_names)
    return DelayedActuator(self, base_actuator)


class DelayedActuator(Actuator):
  """Generic wrapper that adds delay to any actuator.

  Delays the specified command target(s) (position, velocity, and/or effort)
  before passing it to the underlying actuator's compute method.
  """

  def __init__(self, cfg: DelayedActuatorCfg, base_actuator: Actuator) -> None:
    super().__init__(
      base_actuator.entity,
      base_actuator._joint_ids_list,
      base_actuator._joint_names,
    )
    self.cfg = cfg
    self._base_actuator = base_actuator
    self._delay_buffers: dict[str, DelayBuffer] = {}

  def edit_spec(self, spec: mujoco.MjSpec, joint_names: list[str]) -> None:
    self._base_actuator.edit_spec(spec, joint_names)
    self._mjs_actuators = self._base_actuator._mjs_actuators

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    self._base_actuator.initialize(mj_model, model, data, device)

    self._joint_ids = self._base_actuator._joint_ids
    self._ctrl_ids = self._base_actuator._ctrl_ids

    targets = (
      (self.cfg.delay_target,)
      if isinstance(self.cfg.delay_target, str)
      else self.cfg.delay_target
    )

    # Create independent delay buffer for each target.
    for target in targets:
      self._delay_buffers[target] = DelayBuffer(
        min_lag=self.cfg.delay_min_lag,
        max_lag=self.cfg.delay_max_lag,
        batch_size=data.nworld,
        device=device,
        hold_prob=self.cfg.delay_hold_prob,
        update_period=self.cfg.delay_update_period,
        per_env_phase=self.cfg.delay_per_env_phase,
      )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    position_target = cmd.position_target
    velocity_target = cmd.velocity_target
    effort_target = cmd.effort_target

    if "position" in self._delay_buffers:
      self._delay_buffers["position"].append(cmd.position_target)
      position_target = self._delay_buffers["position"].compute()

    if "velocity" in self._delay_buffers:
      self._delay_buffers["velocity"].append(cmd.velocity_target)
      velocity_target = self._delay_buffers["velocity"].compute()

    if "effort" in self._delay_buffers:
      self._delay_buffers["effort"].append(cmd.effort_target)
      effort_target = self._delay_buffers["effort"].compute()

    delayed_cmd = ActuatorCmd(
      position_target=position_target,
      velocity_target=velocity_target,
      effort_target=effort_target,
      joint_pos=cmd.joint_pos,
      joint_vel=cmd.joint_vel,
    )

    return self._base_actuator.compute(delayed_cmd)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    for buffer in self._delay_buffers.values():
      buffer.reset(env_ids)
    self._base_actuator.reset(env_ids)

  def update(self, dt: float) -> None:
    self._base_actuator.update(dt)
