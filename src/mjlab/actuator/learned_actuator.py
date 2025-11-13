"""Learned actuator models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg
from mjlab.utils.buffers import CircularBuffer

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class LearnedMlpActuatorCfg(DcMotorActuatorCfg):
  """Configuration for MLP-based learned actuator model.

  This actuator learns the mapping from joint commands and state history to
  actual torque output. It's useful for capturing actuator dynamics that are
  difficult to model analytically, such as delays, non-linearities, and
  friction effects.

  The network is trained offline using data from the real system and loaded
  as a TorchScript file. The model uses a sliding window of historical joint
  position errors and velocities as inputs.
  """

  network_file: str
  """Path to the TorchScript file containing the trained MLP model."""

  pos_scale: float
  """Scaling factor for joint position error inputs to the network."""

  vel_scale: float
  """Scaling factor for joint velocity inputs to the network."""

  torque_scale: float
  """Scaling factor for torque outputs from the network."""

  input_order: Literal["pos_vel", "vel_pos"] = "pos_vel"
  """Order of inputs to the network.

  - "pos_vel": position errors followed by velocities
  - "vel_pos": velocities followed by position errors
  """

  history_length: int = 3
  """Number of timesteps of history to use as network inputs.

  For example, history_length=3 uses the current timestep plus the previous
  2 timesteps (total of 3 frames).
  """

  # Learned actuators don't use stiffness/damping from PD controller.
  stiffness: float = 0.0
  damping: float = 0.0

  def build(
    self, entity: Entity, joint_ids: list[int], joint_names: list[str]
  ) -> LearnedMlpActuator:
    return LearnedMlpActuator(self, entity, joint_ids, joint_names)


class LearnedMlpActuator(DcMotorActuator[LearnedMlpActuatorCfg]):
  """MLP-based learned actuator with joint history.

  This actuator uses a trained neural network to map from joint commands and
  state history to torque outputs. The network captures complex actuator
  dynamics that are difficult to model analytically.

  The actuator maintains circular buffers of joint position errors and
  velocities, which are used as inputs to the MLP. The network outputs are
  then scaled and clipped using the DC motor limits from the parent class.
  """

  def __init__(
    self,
    cfg: LearnedMlpActuatorCfg,
    entity: Entity,
    joint_ids: list[int],
    joint_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, joint_ids, joint_names)
    self.network: torch.jit.ScriptModule | None = None
    self._pos_error_history: CircularBuffer | None = None
    self._vel_history: CircularBuffer | None = None

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    # Load the trained network from TorchScript file.
    self.network = torch.jit.load(self.cfg.network_file, map_location=device)
    assert self.network is not None
    self.network.eval()

    # Create history buffers.
    num_envs = data.nworld

    self._pos_error_history = CircularBuffer(
      max_len=self.cfg.history_length,
      batch_size=num_envs,
      device=device,
    )
    self._vel_history = CircularBuffer(
      max_len=self.cfg.history_length,
      batch_size=num_envs,
      device=device,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset history buffers for specified environments.

    Args:
      env_ids: Environment indices to reset. If None, reset all environments.
    """
    assert self._pos_error_history is not None
    assert self._vel_history is not None

    if env_ids is None:
      self._pos_error_history.reset()
      self._vel_history.reset()
    elif isinstance(env_ids, slice):
      # Convert slice to indices for CircularBuffer.
      batch_size = self._pos_error_history.batch_size
      indices = list(range(*env_ids.indices(batch_size)))
      self._pos_error_history.reset(indices)
      self._vel_history.reset(indices)
    else:
      self._pos_error_history.reset(env_ids)
      self._vel_history.reset(env_ids)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    """Compute actuator torques using the learned MLP model.

    Args:
      cmd: High-level actuator command containing targets and current state.

    Returns:
      Computed torque tensor of shape (num_envs, num_joints).
    """
    assert self.network is not None
    assert self._pos_error_history is not None
    assert self._vel_history is not None
    assert self._joint_vel_clipped is not None

    # Update history buffers with current state.
    pos_error = cmd.position_target - cmd.joint_pos
    self._pos_error_history.append(pos_error)
    self._vel_history.append(cmd.joint_vel)

    # Save velocity for DC motor clipping in parent class.
    self._joint_vel_clipped[:] = cmd.joint_vel

    num_envs = cmd.joint_pos.shape[0]
    num_joints = cmd.joint_pos.shape[1]

    # Extract history from current to history_length-1 steps back.
    # Each returns shape: (num_envs, num_joints).
    pos_inputs = [
      self._pos_error_history[lag] for lag in range(self.cfg.history_length)
    ]
    vel_inputs = [self._vel_history[lag] for lag in range(self.cfg.history_length)]

    # Stack along feature dimension: (num_envs, num_joints, history_length).
    pos_stacked = torch.stack(pos_inputs, dim=2)
    vel_stacked = torch.stack(vel_inputs, dim=2)

    # Reshape to (num_envs * num_joints, num_history_steps) for network.
    pos_flat = pos_stacked.reshape(num_envs * num_joints, -1)
    vel_flat = vel_stacked.reshape(num_envs * num_joints, -1)

    # Scale and concatenate inputs based on specified order.
    if self.cfg.input_order == "pos_vel":
      network_input = torch.cat(
        [pos_flat * self.cfg.pos_scale, vel_flat * self.cfg.vel_scale], dim=1
      )
    elif self.cfg.input_order == "vel_pos":
      network_input = torch.cat(
        [vel_flat * self.cfg.vel_scale, pos_flat * self.cfg.pos_scale], dim=1
      )
    else:
      raise ValueError(
        f"Invalid input order: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
      )

    # Run network inference.
    with torch.inference_mode():
      torques_flat = self.network(network_input)

    # Reshape and scale output torques.
    computed_torques = torques_flat.reshape(num_envs, num_joints)
    computed_torques = computed_torques * self.cfg.torque_scale

    # Clip using DC motor limits from parent class.
    return self._clip_effort(computed_torques)
