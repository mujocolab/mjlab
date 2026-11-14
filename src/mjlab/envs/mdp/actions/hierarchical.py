"""Hierarchical RL action terms for command routing and frozen low-level policies."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tyro

from mjlab.managers.action_manager import ActionManager, ActionTerm, ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationManager

try:
  import onnxruntime as ort
except ImportError:  # pragma: no cover
  ort = None

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_LOW_LEVEL_OBS_GROUP = "low_level"


def _resolve_param_tensor(
  value: float | tuple[float, ...] | list[float],
  action_dim: int,
  device: str,
  name: str,
) -> float | torch.Tensor:
  """Validate scalar or per-dimension sequence parameter for affine command scaling."""
  if isinstance(value, (float, int)):
    return float(value)
  if isinstance(value, (tuple, list)):
    if len(value) != action_dim:
      raise ValueError(
        f"Expected '{name}' length {action_dim} to match action_dim, got {len(value)}."
      )
    return torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
  raise TypeError(
    f"Unsupported type for '{name}': {type(value)}. "
    "Expected float or tuple[float, ...]."
  )


def _resolve_bound_tensor(
  value: float | tuple[float, ...] | list[float] | None,
  action_dim: int,
  device: str,
  name: str,
) -> torch.Tensor | None:
  """Validate optional scalar or per-dimension clamp bound as a device tensor."""
  if value is None:
    return None
  resolved = _resolve_param_tensor(value, action_dim, device, name)
  if isinstance(resolved, (int, float)):
    return torch.tensor(float(resolved), dtype=torch.float32, device=device)
  return resolved


class _OnnxPolicyWrapper:
  """Callable wrapper around an ONNX Runtime InferenceSession."""

  def __init__(self, policy_path: Path, device: str) -> None:
    if ort is None:
      raise ImportError(
        "onnxruntime is required to load '.onnx' policies in PreTrainedPolicyAction."
      )
    providers = ["CPUExecutionProvider"]
    if (
      device.startswith("cuda")
      and "CUDAExecutionProvider" in ort.get_available_providers()
    ):
      providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    self._session = ort.InferenceSession(str(policy_path), providers=providers)
    inputs = self._session.get_inputs()
    if not inputs:
      raise ValueError(f"ONNX model '{policy_path}' has no inputs.")
    self._input_name = inputs[0].name
    input_shape = inputs[0].shape
    self._fixed_single_batch = bool(input_shape and input_shape[0] == 1)

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    obs_np = obs.detach().cpu().numpy().astype(np.float32, copy=False)
    if self._fixed_single_batch and obs_np.shape[0] > 1:
      chunks = [
        self._session.run(None, {self._input_name: obs_np[i : i + 1]})[0]
        for i in range(obs_np.shape[0])
      ]
      out_np = np.concatenate(chunks, axis=0)
    else:
      out_np = self._session.run(None, {self._input_name: obs_np})[0]
    return torch.from_numpy(np.asarray(out_np, dtype=np.float32)).to(device=obs.device)


class _LowLevelCommandManagerView:
  """Command manager proxy exposing the hierarchical command to low-level obs."""

  def __init__(
    self,
    parent_command_manager: Any,
    command_name: str | None,
    command_buffer: torch.Tensor,
  ) -> None:
    self._parent = parent_command_manager
    self._command_name = command_name
    self._command_buffer = command_buffer

  def get_command(self, name: str) -> torch.Tensor | None:
    if self._command_name is not None and name == self._command_name:
      return self._command_buffer
    return self._parent.get_command(name)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._parent, name)


class _LowLevelEnvView:
  """Environment view routing ``action_manager`` and commands to low-level terms."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str | None,
    command_buffer: torch.Tensor,
  ) -> None:
    self._parent_env = env
    self._command_buffer = command_buffer
    self.command_manager = _LowLevelCommandManagerView(
      env.command_manager, command_name, command_buffer
    )
    self.action_manager: ActionManager | None = None

  @property
  def hierarchical_command(self) -> torch.Tensor:
    """Processed command tensor produced by the enclosing PreTrainedPolicyAction."""
    return self._command_buffer

  def __getattr__(self, name: str) -> Any:
    return getattr(self._parent_env, name)


@dataclass(kw_only=True)
class CommandActionCfg(ActionTermCfg):
  """Configuration for an action term that writes scaled commands into a buffer.

  Useful in hierarchical setups where a high-level policy outputs task commands
  (e.g., base velocity ``(vx, vy, wz)`` or goal pose) consumed by command terms
  or downstream observations.
  """

  entity_name: str = "robot"
  """Name of the associated entity in the scene."""

  action_dim: int | None = None
  """Dimension of the command action. Required when ``command_name`` is None."""

  command_name: str | None = None
  """Optional command term name in ``env.command_manager`` whose command tensor
  is overwritten each step with the processed action."""

  scale: float | tuple[float, ...] = 1.0
  """Scalar or per-dimension multiplier applied to raw actions."""

  offset: float | tuple[float, ...] = 0.0
  """Scalar or per-dimension offset added after scaling."""

  low: float | tuple[float, ...] | None = None
  """Optional lower clamp bound(s) applied to the processed command."""

  high: float | tuple[float, ...] | None = None
  """Optional upper clamp bound(s) applied to the processed command."""

  def build(self, env: ManagerBasedRlEnv) -> CommandAction:
    return CommandAction(self, env)


class CommandAction(ActionTerm):
  """Action term that maps raw high-level actions into a command tensor."""

  cfg: CommandActionCfg
  _entity: Entity

  def __init__(self, cfg: CommandActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg=cfg, env=env)

    target_cmd: torch.Tensor | None = None
    if cfg.command_name is not None:
      target_cmd = env.command_manager.get_command(cfg.command_name)
      if target_cmd is None:
        raise ValueError(
          f"Command '{cfg.command_name}' not found in env.command_manager."
        )

    if cfg.action_dim is None:
      if target_cmd is None:
        raise ValueError(
          "CommandActionCfg requires 'action_dim' when 'command_name' is None."
        )
      self._action_dim = int(target_cmd.shape[-1])
    else:
      if cfg.action_dim < 1:
        raise ValueError(f"action_dim must be >= 1, got {cfg.action_dim}.")
      if target_cmd is not None and target_cmd.shape[-1] != cfg.action_dim:
        raise ValueError(
          f"action_dim ({cfg.action_dim}) does not match command "
          f"'{cfg.command_name}' dimension ({target_cmd.shape[-1]})."
        )
      self._action_dim = cfg.action_dim

    self._scale = _resolve_param_tensor(
      cfg.scale, self._action_dim, self.device, "scale"
    )
    self._offset = _resolve_param_tensor(
      cfg.offset, self._action_dim, self.device, "offset"
    )
    self._low = _resolve_bound_tensor(cfg.low, self._action_dim, self.device, "low")
    self._high = _resolve_bound_tensor(cfg.high, self._action_dim, self.device, "high")

    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._command = torch.zeros_like(self._raw_actions)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def command(self) -> torch.Tensor:
    """Processed command tensor after scale, offset, and bounds clamping."""
    return self._command

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    processed = self._raw_actions * self._scale + self._offset
    if self._low is not None or self._high is not None:
      processed = torch.clamp(processed, min=self._low, max=self._high)
    self._command[:] = processed
    self._write_command_to_manager()

  def apply_actions(self) -> None:
    self._write_command_to_manager()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._command[env_ids] = 0.0

  def _write_command_to_manager(self) -> None:
    if self.cfg.command_name is not None:
      target_cmd = self._env.command_manager.get_command(self.cfg.command_name)
      if target_cmd is not None:
        target_cmd[:] = self._command


@dataclass(kw_only=True)
class PreTrainedPolicyActionCfg(ActionTermCfg):
  """Configuration for a hierarchical RL action term wrapping a frozen policy.

  The high-level policy outputs commands of dimension ``action_dim`` once per
  environment step. Inside the environment's physics decimation loop, this term
  computes ``low_level_observations`` and evaluates the frozen low-level policy
  every ``low_level_decimation`` physics substeps, delegating actuator control
  to ``low_level_actions`` on every physics substep.
  """

  entity_name: str = "robot"
  """Name of the controlled entity in the scene."""

  action_dim: int | None = None
  """Dimension of the high-level command action. Inferred from ``command_name``
  when omitted and ``command_name`` exists in ``env.command_manager``."""

  policy_file: str | None = None
  """Path to a frozen TorchScript (``.pt``/``.jit``) or ONNX (``.onnx``) policy."""

  policy: tyro.conf.Suppress[Callable[[Any], torch.Tensor] | None] = None
  """In-memory policy callable or ``torch.nn.Module`` (mutually exclusive with
  ``policy_file``)."""

  low_level_decimation: int = 1
  """Number of physics substeps per low-level policy inference (>= 1)."""

  low_level_actions: dict[str, ActionTermCfg]
  """Action term configs driven by the low-level policy output."""

  low_level_observations: ObservationGroupCfg
  """Observation group config computed for the low-level policy."""

  command_name: str | None = None
  """Optional command name exposed to ``low_level_observations`` via
  ``mdp.generated_commands(env, command_name=...)``."""

  scale: float | tuple[float, ...] = 1.0
  """Scalar or per-dimension multiplier applied to raw high-level actions."""

  offset: float | tuple[float, ...] = 0.0
  """Scalar or per-dimension offset added to scaled high-level actions."""

  low: float | tuple[float, ...] | None = None
  """Optional lower clamp bound(s) applied to the processed command."""

  high: float | tuple[float, ...] | None = None
  """Optional upper clamp bound(s) applied to the processed command."""

  def build(self, env: ManagerBasedRlEnv) -> PreTrainedPolicyAction:
    return PreTrainedPolicyAction(self, env)


class PreTrainedPolicyAction(ActionTerm):
  """Hierarchical action term driving low-level actions via a frozen policy."""

  cfg: PreTrainedPolicyActionCfg
  _entity: Entity

  def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg=cfg, env=env)

    if (cfg.policy is None) == (cfg.policy_file is None):
      raise ValueError(
        "PreTrainedPolicyActionCfg requires exactly one of 'policy' or 'policy_file'."
      )
    if cfg.low_level_decimation < 1:
      raise ValueError(
        f"low_level_decimation must be >= 1, got {cfg.low_level_decimation}."
      )
    if not cfg.low_level_actions:
      raise ValueError("low_level_actions must contain at least one ActionTermCfg.")

    existing_cmd: torch.Tensor | None = None
    if cfg.command_name is not None:
      existing_cmd = env.command_manager.get_command(cfg.command_name)

    if cfg.action_dim is None:
      if existing_cmd is None:
        raise ValueError(
          "PreTrainedPolicyActionCfg requires 'action_dim' unless 'command_name' "
          "refers to an active term in env.command_manager."
        )
      self._action_dim = int(existing_cmd.shape[-1])
    else:
      if cfg.action_dim < 1:
        raise ValueError(f"action_dim must be >= 1, got {cfg.action_dim}.")
      self._action_dim = cfg.action_dim

    self._scale = _resolve_param_tensor(
      cfg.scale, self._action_dim, self.device, "scale"
    )
    self._offset = _resolve_param_tensor(
      cfg.offset, self._action_dim, self.device, "offset"
    )
    self._low = _resolve_bound_tensor(cfg.low, self._action_dim, self.device, "low")
    self._high = _resolve_bound_tensor(cfg.high, self._action_dim, self.device, "high")

    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._command = torch.zeros_like(self._raw_actions)
    self._substep_counter = 0

    self._policy = self._load_policy(cfg)

    self._low_level_env_view = _LowLevelEnvView(
      env=env,
      command_name=cfg.command_name,
      command_buffer=self._command,
    )
    self._low_level_action_manager = ActionManager(
      cfg.low_level_actions,
      self._low_level_env_view,  # ty: ignore[invalid-argument-type]
    )
    self._low_level_env_view.action_manager = self._low_level_action_manager

    obs_cfg = {_LOW_LEVEL_OBS_GROUP: deepcopy(cfg.low_level_observations)}
    self._low_level_obs_manager = ObservationManager(
      obs_cfg,
      self._low_level_env_view,
    )

  def _load_policy(
    self, cfg: PreTrainedPolicyActionCfg
  ) -> Callable[[Any], torch.Tensor]:
    if cfg.policy is not None:
      if isinstance(cfg.policy, torch.nn.Module):
        cfg.policy.to(self.device)
        cfg.policy.eval()
      return cfg.policy

    assert cfg.policy_file is not None
    policy_path = Path(cfg.policy_file)
    if not policy_path.is_file():
      raise FileNotFoundError(f"Low-level policy file not found: '{policy_path}'.")
    if policy_path.suffix.lower() == ".onnx":
      return _OnnxPolicyWrapper(policy_path, self.device)

    loaded = torch.jit.load(str(policy_path), map_location=self.device)
    loaded.eval()
    return loaded

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def command(self) -> torch.Tensor:
    """Current processed command passed to the low-level policy."""
    return self._command

  @property
  def low_level_actions(self) -> torch.Tensor:
    """Most recent raw actions produced by the low-level policy."""
    return self._low_level_action_manager.action

  @property
  def low_level_action_manager(self) -> ActionManager:
    """Internal ActionManager applying the low-level policy's actions."""
    return self._low_level_action_manager

  @property
  def low_level_observation_manager(self) -> ObservationManager:
    """Internal ObservationManager computing the low-level policy's inputs."""
    return self._low_level_obs_manager

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    processed = self._raw_actions * self._scale + self._offset
    if self._low is not None or self._high is not None:
      processed = torch.clamp(processed, min=self._low, max=self._high)
    self._command[:] = processed
    self._substep_counter = 0

  def apply_actions(self) -> None:
    if self._substep_counter % self.cfg.low_level_decimation == 0:
      if self._substep_counter > 0:
        # Refresh derived kinematics and sensors after intermediate sim.step() calls.
        self._env.sim.forward()
        self._env.sim.sense()
      obs_dict = self._low_level_obs_manager.compute(update_history=True)
      low_level_obs = obs_dict[_LOW_LEVEL_OBS_GROUP]
      with torch.no_grad():
        low_level_action = self._policy(low_level_obs)
      self._low_level_action_manager.process_action(low_level_action)
    self._substep_counter += 1
    self._low_level_action_manager.apply_action()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._command[env_ids] = 0.0
    self._substep_counter = 0
    self._low_level_action_manager.reset(env_ids=env_ids)
    self._low_level_obs_manager.reset(env_ids=env_ids)

    reset_fn = getattr(self._policy, "reset", None)
    if callable(reset_fn):
      dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      dones[env_ids] = True
      if torch.any(dones):
        try:
          reset_fn(dones)
        except TypeError:
          if torch.all(dones):
            reset_fn()
