"""Base actuator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity


class ActuatedType(Enum):
  JOINT = "joint"
  TENDON = "tendon"


@dataclass(kw_only=True)
class ActuatorCfg(ABC):
  actuated_names_expr: tuple[str, ...]
  """Elements (joints/tendons) that are part of this actuator group.

  Can be a tuple of element names or tuple of regex expressions.
  """

  actuated_type: ActuatedType
  """Type of elements being actuated: 'joint' or 'tendon'."""

  armature: float = 0.0
  """Reflected rotor inertia."""

  frictionloss: float = 0.0
  """Friction loss force limit.

  Applies a constant friction force opposing motion, independent of load or velocity.
  Also known as dry friction or load-independent friction.
  """

  @abstractmethod
  def build(
    self, entity: Entity, actuated_ids: list[int], actuated_names: list[str]
  ) -> Actuator:
    """Build actuator instance.

    Args:
      entity: Entity this actuator belongs to.
      actuated_ids: Local element indices (for indexing entity element arrays).
      actuated_names: Element names corresponding to actuated_ids.

    Returns:
      Actuator instance.
    """
    raise NotImplementedError


@dataclass
class ActuatorCmd:
  """High-level actuator command with targets and current state.

  Passed to actuator's `compute()` method to generate low-level control signals.
  All tensors have shape (num_envs, num_joints), except `excitation` which has shape (num_envs, num_tendons).
  """

  position_target: torch.Tensor | None
  """Desired joint positions."""
  velocity_target: torch.Tensor | None
  """Desired joint velocities."""
  effort_target: torch.Tensor | None
  """Feedforward effort."""
  excitation: torch.Tensor | None
  """Muscle excitation signal."""
  joint_pos: torch.Tensor | None
  """Current joint positions."""
  joint_vel: torch.Tensor | None
  """Current joint velocities."""


class Actuator(ABC):
  """Base actuator interface."""

  def __init__(
    self,
    entity: Entity,
    actuated_ids: list[int],
    actuated_names: list[str],
  ) -> None:
    self.entity = entity
    self._actuated_ids_list = actuated_ids
    self._actuated_names = actuated_names
    self._actuated_ids: torch.Tensor | None = None
    self._ctrl_ids: torch.Tensor | None = None
    self._mjs_actuators: list[mujoco.MjsActuator] = []

  @property
  def actuated_ids(self) -> torch.Tensor:
    """Local indices of Actuated elements controlled by this actuator."""
    assert self._actuated_ids is not None
    return self._actuated_ids

  @property
  def actuated_names(self) -> list[str]:
    """Names of actuated elements controlled by this actuator."""
    return self._actuated_names

  @property
  def actuated_type(self) -> ActuatedType:
    """Type of elements controlled by this actuator: 'joint' or 'tendon'."""
    # Infer from the first actuated name.
    first_name = self._actuated_names[0]
    if first_name in self.entity.joint_names:
      return ActuatedType.JOINT
    elif first_name in self.entity.tendon_names:
      return ActuatedType.TENDON
    else:
      raise ValueError(
        f"Actuated name '{first_name}' not found in entity joints or tendons."
      )

  @property
  def ctrl_ids(self) -> torch.Tensor:
    """Global indices of control inputs for this actuator."""
    assert self._ctrl_ids is not None
    return self._ctrl_ids

  @abstractmethod
  def edit_spec(self, spec: mujoco.MjSpec, actuated_names: list[str]) -> None:
    """Edit the MjSpec to add actuators, and configure actuated elements (joints/tendons).

    This is called during entity construction, before the model is compiled.

    Args:
      spec: The entity's MjSpec to edit.
      actuated_names: Names of actuated elements (joints/tendons) controlled by this actuator.
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
    del mj_model, model, data  # Unused.
    self._actuated_ids = torch.tensor(
      self._actuated_ids_list, dtype=torch.long, device=device
    )
    ctrl_ids_list = [act.id for act in self._mjs_actuators]
    self._ctrl_ids = torch.tensor(ctrl_ids_list, dtype=torch.long, device=device)

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
