"""An ideal PD control actuator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
  from mjlab.entity import Entity

IdealPdCfgT = TypeVar("IdealPdCfgT", bound="IdealPdActuatorCfg")


@dataclass(kw_only=True)
class IdealPdActuatorCfg(ActuatorCfg):
  """Configuration for ideal PD actuator."""

  stiffness: float
  """PD stiffness (proportional gain)."""
  damping: float
  """PD damping (derivative gain)."""
  effort_limit: float = float("inf")
  """Maximum force/torque limit."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> IdealPdActuator:
    return IdealPdActuator(self, entity, target_ids, target_names)


class IdealPdActuator(Actuator, Generic[IdealPdCfgT]):
  """Ideal PD control actuator."""

  def __init__(
    self,
    cfg: IdealPdCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.stiffness: torch.Tensor | None = None
    self.damping: torch.Tensor | None = None
    self.force_limit: torch.Tensor | None = None
    self.default_stiffness: torch.Tensor | None = None
    self.default_damping: torch.Tensor | None = None
    self.default_force_limit: torch.Tensor | None = None

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # Add <motor> actuator to spec, one per target.
    for target_name in target_names:
      actuator = create_motor_actuator(
        spec,
        target_name,
        effort_limit=self.cfg.effort_limit,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

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
    self.stiffness = torch.full(
      (num_envs, num_joints), self.cfg.stiffness, dtype=torch.float, device=device
    )
    self.damping = torch.full(
      (num_envs, num_joints), self.cfg.damping, dtype=torch.float, device=device
    )
    self.force_limit = torch.full(
      (num_envs, num_joints), self.cfg.effort_limit, dtype=torch.float, device=device
    )

    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()
    self.default_force_limit = self.force_limit.clone()

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.stiffness is not None
    assert self.damping is not None

    pos_error = cmd.position_target - cmd.pos
    vel_error = cmd.velocity_target - cmd.vel

    computed_torques = self.stiffness * pos_error
    computed_torques += self.damping * vel_error
    computed_torques += cmd.effort_target

    return self._clip_effort(computed_torques)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self.force_limit is not None
    return torch.clamp(effort, -self.force_limit, self.force_limit)

  def set_gains(
    self,
    env_ids: torch.Tensor | slice,
    kp: torch.Tensor | None = None,
    kd: torch.Tensor | None = None,
  ) -> None:
    """Set PD gains for specified environments.

    Args:
      env_ids: Environment indices to update.
      kp: New proportional gains. Shape: (num_envs, num_actuators) or (num_envs,).
      kd: New derivative gains. Shape: (num_envs, num_actuators) or (num_envs,).
    """
    assert self.stiffness is not None
    assert self.damping is not None

    if kp is not None:
      if kp.ndim == 1:
        kp = kp.unsqueeze(-1)
      self.stiffness[env_ids] = kp

    if kd is not None:
      if kd.ndim == 1:
        kd = kd.unsqueeze(-1)
      self.damping[env_ids] = kd

  def set_effort_limit(
    self, env_ids: torch.Tensor | slice, effort_limit: torch.Tensor
  ) -> None:
    """Set effort limits for specified environments.

    Args:
      env_ids: Environment indices to update.
      effort_limit: New effort limits. Shape: (num_envs, num_actuators) or (num_envs,).
    """
    assert self.force_limit is not None

    if effort_limit.ndim == 1:
      effort_limit = effort_limit.unsqueeze(-1)
    self.force_limit[env_ids] = effort_limit
