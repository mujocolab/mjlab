"""Base actuator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class ActuatorCfg(ABC):
  joint_names_expr: tuple[str, ...]
  """Joints that are part of this actuator group.

  Can be a tuple of joint names or tuple of regex expressions.
  """

  armature: float = 0.0
  """Reflected rotor inertia."""

  frictionloss: float = 0.0
  """Friction loss force limit.

  Applies a constant friction force opposing motion, independent of load or velocity.
  Also known as dry friction or load-independent friction.
  """

  @abstractmethod
  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> Actuator:
    """Build actuator instance.

    Args:
      entity: Entity this actuator belongs to.
      joint_ids: Local joint indices (for indexing entity joint arrays).
      joint_names: Joint names corresponding to joint_ids.

    Returns:
      Actuator instance.
    """
    raise NotImplementedError


@dataclass
class ActuatorCmd:
  """High-level actuator command with targets and current state.

  Passed to actuator's `compute()` method to generate low-level control signals.
  All tensors have shape (num_envs, num_joints).
  """

  position_target: torch.Tensor
  """Desired joint positions."""
  velocity_target: torch.Tensor
  """Desired joint velocities."""
  effort_target: torch.Tensor
  """Feedforward effort."""
  joint_pos: torch.Tensor
  """Current joint positions."""
  joint_vel: torch.Tensor
  """Current joint velocities."""


class Actuator(ABC):
  """Base actuator interface."""

  def __init__(
    self,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    self.entity = entity
    self._joint_ids_list = joint_ids
    self._joint_names = joint_names
    self._joint_ids: torch.Tensor | None = None
    self._ctrl_ids: torch.Tensor | None = None
    self._mjs_actuators: list[mujoco.MjsActuator] = []

  @property
  def joint_ids(self) -> torch.Tensor:
    """Local indices of joints controlled by this actuator."""
    assert self._joint_ids is not None
    return self._joint_ids

  @property
  def ctrl_ids(self) -> torch.Tensor:
    """Global indices of control inputs for this actuator."""
    assert self._ctrl_ids is not None
    return self._ctrl_ids

  @abstractmethod
  def edit_spec(self, spec: mujoco.MjSpec, joint_names: list[str]) -> None:
    """Edit the MjSpec to add actuators and configure joints.

    This is called during entity construction, before the model is compiled.

    Args:
      spec: The entity's MjSpec to edit.
      joint_names: Names of joints controlled by this actuator.
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
    self._joint_ids = torch.tensor(
      self._joint_ids_list, dtype=torch.long, device=device
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
