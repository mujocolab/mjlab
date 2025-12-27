"""Base actuator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.entity.data import EntityData

ActuatorCfgT = TypeVar("ActuatorCfgT", bound="ActuatorCfg")


class TransmissionType(str, Enum):
  """Transmission types for actuators."""

  JOINT = "joint"
  TENDON = "tendon"
  SITE = "site"


@dataclass(kw_only=True)
class ActuatorCfg(ABC):
  target_names_expr: tuple[str, ...]
  """Targets that are part of this actuator group.

  Can be a tuple of names or tuple of regex expressions.
  Interpreted based on transmission_type (joint/tendon/site).
  """

  transmission_type: TransmissionType = TransmissionType.JOINT
  """Transmission type. Defaults to JOINT."""

  armature: float = 0.0
  """Reflected rotor inertia."""

  frictionloss: float = 0.0
  """Friction loss force limit.

  Applies a constant friction force opposing motion, independent of load or velocity.
  Also known as dry friction or load-independent friction.
  """

  def __post_init__(self) -> None:
    assert self.armature >= 0.0, "armature must be non-negative."
    assert self.frictionloss >= 0.0, "frictionloss must be non-negative."
    if self.transmission_type == TransmissionType.SITE:
      if self.armature > 0.0 or self.frictionloss > 0.0:
        raise ValueError(
          f"{self.__class__.__name__}: armature and frictionloss are not supported for "
          "SITE transmission type."
        )

  @abstractmethod
  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> Actuator:
    """Build actuator instance.

    Args:
      entity: Entity this actuator belongs to.
      target_ids: Local target indices (for indexing entity arrays).
      target_names: Target names corresponding to target_ids.

    Returns:
      Actuator instance.
    """
    raise NotImplementedError


@dataclass
class ActuatorCmd:
  """High-level actuator command with targets and current state.

  Passed to actuator's `compute()` method to generate low-level control signals.
  All tensors have shape (num_envs, num_targets).
  """

  position_target: torch.Tensor
  """Desired positions (joint positions, tendon lengths, or site positions)."""
  velocity_target: torch.Tensor
  """Desired velocities (joint velocities, tendon velocities, or site velocities)."""
  effort_target: torch.Tensor
  """Feedforward effort (torques or forces)."""
  pos: torch.Tensor
  """Current positions (joint positions, tendon lengths, or site positions)."""
  vel: torch.Tensor
  """Current velocities (joint velocities, tendon velocities, or site velocities)."""


class Actuator(ABC, Generic[ActuatorCfgT]):
  """Base actuator interface."""

  def __init__(
    self,
    cfg: ActuatorCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    self.cfg = cfg
    self.entity = entity
    self._target_ids_list = target_ids
    self._target_names = target_names
    self._target_ids: torch.Tensor | None = None
    self._ctrl_ids: torch.Tensor | None = None
    self._mjs_actuators: list[mujoco.MjsActuator] = []
    self._site_zeros: torch.Tensor | None = None

  @property
  def target_ids(self) -> torch.Tensor:
    """Local indices of targets controlled by this actuator."""
    assert self._target_ids is not None
    return self._target_ids

  @property
  def target_names(self) -> list[str]:
    """Names of targets controlled by this actuator."""
    return self._target_names

  @property
  def transmission_type(self) -> TransmissionType:
    """Transmission type of this actuator."""
    return self.cfg.transmission_type

  @property
  def ctrl_ids(self) -> torch.Tensor:
    """Global indices of control inputs for this actuator."""
    assert self._ctrl_ids is not None
    return self._ctrl_ids

  @abstractmethod
  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    """Edit the MjSpec to add actuators.

    This is called during entity construction, before the model is compiled.

    Args:
      spec: The entity's MjSpec to edit.
      target_names: Names of targets (joints, tendons, or sites) controlled by
        this actuator.
    """
    raise NotImplementedError

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    """Initialize the actuator after model compilation.

    This is called after the MjSpec is compiled into an MjModel.

    Args:
      mj_model: The compiled MuJoCo model.
      model: The compiled mjwarp model.
      data: The mjwarp data arrays.
      device: Device for tensor operations (e.g., "cuda", "cpu").
    """
    del mj_model, model  # Unused.
    self._target_ids = torch.tensor(
      self._target_ids_list, dtype=torch.long, device=device
    )
    ctrl_ids_list = [act.id for act in self._mjs_actuators]
    self._ctrl_ids = torch.tensor(ctrl_ids_list, dtype=torch.long, device=device)

    # Pre-allocate zeros for SITE transmission type to avoid repeated allocations.
    if self.transmission_type == TransmissionType.SITE:
      nenvs = data.nworld
      ntargets = len(self._target_ids_list)
      self._site_zeros = torch.zeros((nenvs, ntargets), device=device)

  def get_command(self, data: EntityData) -> ActuatorCmd:
    """Extract command data for this actuator from entity data.

    Args:
      data: The entity data containing all state and target information.

    Returns:
      ActuatorCmd with appropriate data based on transmission type.
    """
    if self.transmission_type == TransmissionType.JOINT:
      return ActuatorCmd(
        position_target=data.joint_pos_target[:, self.target_ids],
        velocity_target=data.joint_vel_target[:, self.target_ids],
        effort_target=data.joint_effort_target[:, self.target_ids],
        pos=data.joint_pos[:, self.target_ids],
        vel=data.joint_vel[:, self.target_ids],
      )
    elif self.transmission_type == TransmissionType.TENDON:
      return ActuatorCmd(
        position_target=data.tendon_len_target[:, self.target_ids],
        velocity_target=data.tendon_vel_target[:, self.target_ids],
        effort_target=data.tendon_effort_target[:, self.target_ids],
        pos=data.tendon_len[:, self.target_ids],
        vel=data.tendon_vel[:, self.target_ids],
      )
    elif self.transmission_type == TransmissionType.SITE:
      assert self._site_zeros is not None
      return ActuatorCmd(
        position_target=self._site_zeros,
        velocity_target=self._site_zeros,
        effort_target=data.site_effort_target[:, self.target_ids],
        pos=self._site_zeros,
        vel=self._site_zeros,
      )
    else:
      raise ValueError(f"Unknown transmission type: {self.transmission_type}")

  @abstractmethod
  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    """Compute low-level actuator control signal from high-level commands.

    Args:
      cmd: High-level actuator command.

    Returns:
      Control signal tensor of shape (num_envs, num_actuators).
    """
    raise NotImplementedError

  # Optional methods.

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset actuator state for specified environments.

    Base implementation does nothing. Override in subclasses that maintain
    internal state.

    Args:
      env_ids: Environment indices to reset. If None, reset all environments.
    """
    del env_ids  # Unused.

  def update(self, dt: float) -> None:
    """Update actuator state after a simulation step.

    Base implementation does nothing. Override in subclasses that need
    per-step updates.

    Args:
      dt: Time step in seconds.
    """
    del dt  # Unused.
