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
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque

if TYPE_CHECKING:
  from mjlab.entity import Entity

DcMotorCfgT = TypeVar("DcMotorCfgT", bound="DcMotorActuatorCfg")


def dc_motor_clip(
  effort: torch.Tensor,
  saturation_effort: torch.Tensor,
  velocity_limit: torch.Tensor,
  force_limit: torch.Tensor,
  vel: torch.Tensor,
) -> torch.Tensor:
  """Clip effort to the DC motor torque-speed curve.

  Linear torque-speed curve: full saturation_effort at zero velocity falling to
  zero at velocity_limit, further bounded by the continuous force_limit. Shared
  by DcMotorActuator._clip_effort and the fused control law so the curve lives in
  one place.

  The corner velocity where the curve intersects force_limit is recomputed from
  force_limit every call rather than cached, since force_limit can be
  domain-randomized after initialize() (e.g. via set_effort_limit); caching it
  would silently go stale.
  """
  vel_at_effort_lim = velocity_limit * (1 + force_limit / saturation_effort)
  vel_clipped = torch.clamp(vel, min=-vel_at_effort_lim, max=vel_at_effort_lim)
  torque_speed_top = saturation_effort * (1.0 - vel_clipped / velocity_limit)
  torque_speed_bottom = saturation_effort * (-1.0 - vel_clipped / velocity_limit)
  max_effort = torch.clamp(torque_speed_top, max=force_limit)
  min_effort = torch.clamp(torque_speed_bottom, min=-force_limit)
  return torch.clamp(effort, min=min_effort, max=max_effort)


@dataclass(kw_only=True)
class DcMotorActuatorCfg(IdealPdActuatorCfg):
  """Configuration for DC motor actuator with velocity-based saturation.

  This actuator implements a DC motor torque-speed curve for more realistic
  actuator behavior. The motor produces maximum torque (saturation_effort) at
  zero velocity and reduces linearly to zero torque at maximum velocity.

  Note: effort_limit should be explicitly set to a realistic value for proper
  motor modeling. Using the default (inf) will trigger a warning. Use
  IdealPdActuator if unlimited torque is desired.

  For a native MuJoCo ``<dcmotor>`` with back-EMF, voltage saturation, and
  configurable ``Kt`` / ``Ke`` / ``R``, see ``BuiltinDcMotorActuator``.
  """

  saturation_effort: float
  """Peak motor torque at zero velocity (stall torque)."""

  velocity_limit: float
  """Maximum motor velocity (no-load speed)."""

  def __post_init__(self) -> None:
    """Validate DC motor parameters."""
    super().__post_init__()
    import warnings

    if self.effort_limit == float("inf"):
      warnings.warn(
        "effort_limit is set to inf for DcMotorActuator, which creates an "
        "unrealistic motor with unlimited continuous torque. Consider setting "
        "effort_limit to your motor's continuous rating (<= saturation_effort). "
        "Use IdealPdActuator if you truly want unlimited torque.",
        UserWarning,
        stacklevel=2,
      )

    if self.effort_limit > self.saturation_effort:
      warnings.warn(
        f"effort_limit ({self.effort_limit}) exceeds saturation_effort "
        f"({self.saturation_effort}). For realistic motors, continuous torque "
        "should be <= peak torque.",
        UserWarning,
        stacklevel=2,
      )

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> DcMotorActuator:
    return DcMotorActuator(self, entity, target_ids, target_names)


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

  # Same stateless-law compute as IdealPd, with the torque-speed curve added to
  # the law. The velocity feedback the curve needs comes straight from cmd.vel.
  param_names = (
    *IdealPdActuator.param_names,
    "saturation_effort",
    "velocity_limit_motor",
  )

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    torque = pd_torque(params["stiffness"], params["damping"], cmd)
    return dc_motor_clip(
      torque,
      params["saturation_effort"],
      params["velocity_limit_motor"],
      params["force_limit"],
      cmd.vel,
    )

  def __init__(
    self,
    cfg: DcMotorCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.saturation_effort: torch.Tensor | None = None
    self.velocity_limit_motor: torch.Tensor | None = None
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
    num_joints = len(self._target_names)

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
    self._joint_vel_clipped = torch.zeros(num_envs, num_joints, device=device)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    # Retained for LearnedMlpActuator, which has its own (stateful) compute and
    # reuses this torque-speed clip. DcMotorActuator itself clips inside
    # control_law via dc_motor_clip.
    assert self.saturation_effort is not None
    assert self.velocity_limit_motor is not None
    assert self.force_limit is not None
    assert self._joint_vel_clipped is not None
    return dc_motor_clip(
      effort,
      self.saturation_effort,
      self.velocity_limit_motor,
      self.force_limit,
      self._joint_vel_clipped,
    )
