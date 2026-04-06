"""Base actuator interface."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.entity.data import EntityData

ActuatorCfgT = TypeVar("ActuatorCfgT", bound="ActuatorCfg")

CommandField = Literal["position", "velocity", "effort"]


class TransmissionType(str, Enum):
  """Transmission types for actuators."""

  JOINT = "joint"
  TENDON = "tendon"
  SITE = "site"


@dataclass(kw_only=True)
class ActuatorCfg(ABC):
  target_names_expr: tuple[str, ...]
  """Targets that are part of this actuator group.

  Can be a tuple of names or tuple of regex expressions. Interpreted based on
  transmission_type.
  """

  transmission_type: TransmissionType = TransmissionType.JOINT
  """Transmission type. Defaults to JOINT."""

  armature: float | None = None
  """Reflected rotor inertia. None preserves the XML value."""

  frictionloss: float | None = None
  """Friction loss force limit. None preserves the XML value.

  Applies a constant friction force opposing motion, independent of load or velocity.
  Also known as dry friction or load-independent friction.
  """

  viscous_damping: float | None = None
  """Passive viscous damping coefficient. None preserves the XML value.

  Produces a dissipative force f(v) = -b·v proportional to velocity. Always present
  regardless of actuator activity. Unlike ``damping`` (the PD derivative gain kv, which
  is active control), this is a passive property.

  Maps to ``<joint damping>`` for JOINT transmission and ``<tendon damping>``
  for TENDON transmission. Ignored for SITE.
  """

  delay_min_lag: int = 0
  """Minimum command delay in physics timesteps.

  Each step, a lag is sampled uniformly from [min, max]. The command target arrives
  that many steps late at the actuator's control law. Models communication and bus
  latency between the policy and the motor (as opposed to observation delay, which
  models sensor pipeline latency).
  """

  delay_max_lag: int = 0
  """Maximum command delay in physics timesteps. Set > 0 to enable delay."""

  delay_hold_prob: float = 0.0
  """Probability of keeping the current lag instead of resampling."""

  delay_update_period: int = 0
  """How often to resample the lag, in physics timesteps (0 = every step)."""

  delay_per_env_phase: bool = True
  """Stagger lag resampling across environments so they don't all update
  on the same step."""

  def __post_init__(self) -> None:
    if self.armature is not None:
      assert self.armature >= 0.0, "armature must be non-negative."
    if self.frictionloss is not None:
      assert self.frictionloss >= 0.0, "frictionloss must be non-negative."
    if self.viscous_damping is not None:
      assert self.viscous_damping >= 0.0, "viscous_damping must be non-negative."
    if self.transmission_type == TransmissionType.SITE:
      if (
        (self.armature is not None and self.armature > 0.0)
        or (self.frictionloss is not None and self.frictionloss > 0.0)
        or (self.viscous_damping is not None and self.viscous_damping > 0.0)
      ):
        raise ValueError(
          f"{self.__class__.__name__}: armature, frictionloss, and viscous_damping are "
          "not supported for SITE transmission type."
        )
    assert self.delay_min_lag >= 0, "delay_min_lag must be non-negative."
    assert self.delay_max_lag >= 0, "delay_max_lag must be non-negative."
    assert self.delay_min_lag <= self.delay_max_lag, (
      "delay_min_lag must be <= delay_max_lag."
    )
    assert 0.0 <= self.delay_hold_prob <= 1.0, "delay_hold_prob must be in [0, 1]."
    assert self.delay_update_period >= 0, "delay_update_period must be non-negative."

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
    self._global_ctrl_ids: torch.Tensor | None = None
    self._mjs_actuators: list[mujoco.MjsActuator] = []
    self._site_zeros: torch.Tensor | None = None
    self._delay_buffer: DelayBuffer | None = None

  @property
  def has_delay(self) -> bool:
    """Whether this actuator has delay configured."""
    return self.cfg.delay_max_lag > 0

  @property
  def command_field(self) -> CommandField | None:
    """The primary command field this actuator consumes.

    Returns None by default. Subclasses should override to return the
    appropriate field.
    """
    return None

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
    """Local indices of control inputs within the entity."""
    assert self._ctrl_ids is not None
    return self._ctrl_ids

  @property
  def global_ctrl_ids(self) -> torch.Tensor:
    """Global indices of control inputs in the MuJoCo model."""
    assert self._global_ctrl_ids is not None
    return self._global_ctrl_ids

  @abstractmethod
  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    """Edit the MjSpec to add actuators.

    This is called during entity construction, before the model is compiled.

    Args:
      spec: The entity's MjSpec to edit.
      target_names: Names of targets (joints, tendons, or sites) as they
        appear in the spec. When the entity's ``spec_fn`` uses internal
        ``MjSpec.attach(prefix=...)``, these will include the prefix
        (e.g., ``"left/elbow"`` rather than ``"elbow"``).
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
    global_ctrl_ids_list = [act.id for act in self._mjs_actuators]
    self._global_ctrl_ids = torch.tensor(
      global_ctrl_ids_list, dtype=torch.long, device=device
    )
    entity_ctrl_ids = self.entity.indexing.ctrl_ids
    global_to_local = {gid.item(): i for i, gid in enumerate(entity_ctrl_ids)}
    self._ctrl_ids = torch.tensor(
      [global_to_local[gid] for gid in global_ctrl_ids_list],
      dtype=torch.long,
      device=device,
    )

    # Pre-allocate zeros for SITE transmission type to avoid repeated allocations.
    if self.transmission_type == TransmissionType.SITE:
      nenvs = data.nworld
      ntargets = len(self._target_ids_list)
      self._site_zeros = torch.zeros((nenvs, ntargets), device=device)

    self._init_delay_buffer(data.nworld, device)

  def _init_delay_buffer(self, num_envs: int, device: str) -> None:
    """Create delay buffer. Called during initialize()."""
    if not self.has_delay:
      return
    if self.command_field is None:
      raise ValueError(
        f"{self.__class__.__name__}: delay is configured (delay_max_lag="
        f"{self.cfg.delay_max_lag}) but command_field is not defined."
      )
    self._delay_buffer = DelayBuffer(
      min_lag=self.cfg.delay_min_lag,
      max_lag=self.cfg.delay_max_lag,
      batch_size=num_envs,
      device=device,
      hold_prob=self.cfg.delay_hold_prob,
      update_period=self.cfg.delay_update_period,
      per_env_phase=self.cfg.delay_per_env_phase,
    )

  def apply_delay(self, cmd: ActuatorCmd) -> ActuatorCmd:
    """Apply delay to the command_field target. No-op without delay."""
    if self._delay_buffer is None:
      return cmd
    cf = self.command_field
    if cf == "position":
      self._delay_buffer.append(cmd.position_target)
      return dataclasses.replace(cmd, position_target=self._delay_buffer.compute())
    elif cf == "velocity":
      self._delay_buffer.append(cmd.velocity_target)
      return dataclasses.replace(cmd, velocity_target=self._delay_buffer.compute())
    else:
      self._delay_buffer.append(cmd.effort_target)
      return dataclasses.replace(cmd, effort_target=self._delay_buffer.compute())

  def set_lags(
    self,
    lags: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """Set delay lag values for specified environments.

    Built-in actuators with the same delay config share a fused delay
    buffer for performance. Calling ``set_lags`` on any one of them
    affects the entire fused group.

    Args:
      lags: Lag values in physics timesteps. Shape: (num_env_ids,) or scalar.
      env_ids: Environment indices to set. If None, sets all environments.
    """
    if self._delay_buffer is not None:
      self._delay_buffer.set_lags(lags, env_ids)

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

    Resets delay buffers if present. Subclasses that override this should
    call ``super().reset(env_ids)``.

    Args:
      env_ids: Environment indices to reset. If None, reset all environments.
    """
    if self._delay_buffer is not None:
      self._delay_buffer.reset(env_ids)

  def update(self, dt: float) -> None:
    """Update actuator state after a simulation step.

    Base implementation does nothing. Override in subclasses that need
    per-step updates.

    Args:
      dt: Time step in seconds.
    """
    del dt  # Unused.
