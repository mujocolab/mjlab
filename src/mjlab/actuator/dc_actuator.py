"""DC motor actuator with velocity-based saturation model.

This module provides a DC motor actuator that implements a realistic torque-speed
curve for more accurate motor behavior simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity

DcMotorCfgT = TypeVar("DcMotorCfgT", bound="DcMotorActuatorCfg")


@dataclass(kw_only=True)
class DcMotorActuatorCfg(IdealPdActuatorCfg):
  """Configuration for DC motor actuator with velocity-based saturation.

  This actuator implements a DC motor torque-speed curve for more realistic
  actuator behavior. The motor produces maximum torque (saturation_effort) at
  zero velocity and reduces linearly to zero torque at maximum velocity.
  """

  saturation_effort: float
  """Peak motor torque at zero velocity (stall torque)."""

  velocity_limit: float
  """Maximum motor velocity (no-load speed)."""

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> DcMotorActuator:
    return DcMotorActuator(self, entity, joint_ids, joint_names)


class DcMotorActuator(IdealPdActuator[DcMotorCfgT], Generic[DcMotorCfgT]):
  """DC motor actuator with velocity-based saturation model.

  This actuator extends IdealPdActuator with a realistic DC motor model
  that limits torque based on current joint velocity. The model implements
  a linear torque-speed curve where:
  - At zero velocity: can produce full saturation_effort (stall torque)
  - At max velocity: can produce zero torque
  - Between: torque limit varies linearly

  The continuous torque limit (effort_limit) further constrains the output.
  """

  def __init__(
    self,
    cfg: DcMotorCfgT,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, joint_ids, joint_names)
    self.saturation_effort: torch.Tensor | None = None
    self.velocity_limit_motor: torch.Tensor | None = None
    self._vel_at_effort_lim: torch.Tensor | None = None
    self._joint_vel_clipped: torch.Tensor | None = None

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    num_envs = data.nworld
    num_joints = len(self._joint_names)

    self.saturation_effort = torch.full(
      (num_envs, num_joints),
      self.cfg.saturation_effort,
      dtype=torch.float,
      device=device,
    )
    self.velocity_limit_motor = torch.full(
      (num_envs, num_joints),
      self.cfg.velocity_limit,
      dtype=torch.float,
      device=device,
    )

    # Compute corner velocity where torque-speed curve intersects effort_limit.
    assert self.force_limit is not None
    self._vel_at_effort_lim = self.velocity_limit_motor * (
      1 + self.force_limit / self.saturation_effort
    )
    self._joint_vel_clipped = torch.zeros(num_envs, num_joints, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._joint_vel_clipped is not None
    self._joint_vel_clipped[:] = cmd.joint_vel
    return super().compute(cmd)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self.saturation_effort is not None
    assert self.velocity_limit_motor is not None
    assert self.force_limit is not None
    assert self._vel_at_effort_lim is not None
    assert self._joint_vel_clipped is not None

    # Clip velocity to corner velocity range.
    vel_clipped = torch.clamp(
      self._joint_vel_clipped,
      min=-self._vel_at_effort_lim,
      max=self._vel_at_effort_lim,
    )

    # Compute torque-speed curve limits.
    torque_speed_top = self.saturation_effort * (
      1.0 - vel_clipped / self.velocity_limit_motor
    )
    torque_speed_bottom = self.saturation_effort * (
      -1.0 - vel_clipped / self.velocity_limit_motor
    )

    # Apply continuous torque constraint.
    max_effort = torch.clamp(torque_speed_top, max=self.force_limit)
    min_effort = torch.clamp(torque_speed_bottom, min=-self.force_limit)

    return torch.clamp(effort, min=min_effort, max=max_effort)
