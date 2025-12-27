"""Actions that control actuator transmissions (e.g., joints, tendons, sites)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BaseActionCfg(ActionTermCfg):
  """Configuration for actions that control actuator transmissions."""

  transmission_type: TransmissionType = TransmissionType.JOINT
  """Type of transmission to control."""

  actuator_names: tuple[str, ...] | list[str]
  """Actuator names to control."""

  scale: float | dict[str, float] = 1.0
  """Action scale. Float or dict mapping actuator names to scales."""

  offset: float | dict[str, float] = 0.0
  """Action offset. Float or dict mapping actuator names to offsets."""

  preserve_order: bool = False
  """Whether to preserve the order of actuator names."""


class BaseAction(ActionTerm):
  """Apply actions to actuator transmissions with scale/offset processing.

  Supports controlling different transmission types (e.g., joints, tendons,
  sites) with configurable affine transformations applied to raw actions.
  """

  cfg: BaseActionCfg
  _entity: Entity

  def __init__(self, cfg: BaseActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    # Find targets based on transmission type.
    target_ids, target_names = self._find_targets(cfg)
    self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
    self._target_names = target_names

    self._num_targets = len(target_ids)
    self._action_dim = len(target_ids)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    if isinstance(cfg.scale, (float, int)):
      self._scale = float(cfg.scale)
    elif isinstance(cfg.scale, dict):
      self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.scale, self._target_names
      )
      self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported scale type: {type(cfg.scale)}. "
        f"Supported types are float and dict."
      )

    if isinstance(cfg.offset, (float, int)):
      self._offset = float(cfg.offset)
    elif isinstance(cfg.offset, dict):
      self._offset = torch.zeros_like(self._raw_actions)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.offset, self._target_names
      )
      self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported offset type: {type(cfg.offset)}. "
        f"Supported types are float and dict."
      )

  def _find_targets(self, cfg: BaseActionCfg) -> tuple[list[int], list[str]]:
    """Find target IDs and names based on transmission type.

    Args:
      cfg: Action configuration.

    Returns:
      Tuple of (target_ids, target_names).
    """
    if cfg.transmission_type == TransmissionType.JOINT:
      return self._entity.find_joints_by_actuator_names(cfg.actuator_names)
    elif cfg.transmission_type == TransmissionType.TENDON:
      return self._entity.find_tendons(
        cfg.actuator_names, preserve_order=cfg.preserve_order
      )
    elif cfg.transmission_type == TransmissionType.SITE:
      return self._entity.find_sites(
        cfg.actuator_names, preserve_order=cfg.preserve_order
      )
    else:
      raise ValueError(f"Unknown transmission type: {cfg.transmission_type}")

  # Properties.

  @property
  def scale(self) -> torch.Tensor | float:
    """Action scale."""
    return self._scale

  @property
  def offset(self) -> torch.Tensor | float:
    """Action offset."""
    return self._offset

  @property
  def raw_action(self) -> torch.Tensor:
    """Raw actions (before scale/offset)."""
    return self._raw_actions

  @property
  def action_dim(self) -> int:
    """Dimension of the action space."""
    return self._action_dim

  @property
  def target_ids(self) -> torch.Tensor:
    """Target IDs for the controlled transmission."""
    return self._target_ids

  @property
  def target_names(self) -> list[str]:
    """Target names for the controlled transmission."""
    return self._target_names

  def process_actions(self, actions: torch.Tensor):
    """Process raw actions by applying scale and offset."""
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset raw actions to zero for specified environments."""
    self._raw_actions[env_ids] = 0.0


##
# Joint actions.
##


@dataclass(kw_only=True)
class JointPositionActionCfg(BaseActionCfg):
  """Configuration for joint position control."""

  use_default_offset: bool = True

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointPositionAction:
    return JointPositionAction(self, env)


@dataclass(kw_only=True)
class JointVelocityActionCfg(BaseActionCfg):
  """Configuration for joint velocity control."""

  use_default_offset: bool = True

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointVelocityAction:
    return JointVelocityAction(self, env)


@dataclass(kw_only=True)
class JointEffortActionCfg(BaseActionCfg):
  """Configuration for joint effort (torque) control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointEffortAction:
    return JointEffortAction(self, env)


class JointPositionAction(BaseAction):
  """Control joints via position targets."""

  def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


class JointVelocityAction(BaseAction):
  """Control joints via velocity targets."""

  def __init__(self, cfg: JointVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_vel[:, self._target_ids].clone()

  def apply_actions(self) -> None:
    self._entity.set_joint_velocity_target(
      self._processed_actions, joint_ids=self._target_ids
    )


class JointEffortAction(BaseAction):
  """Control joints via effort (torque) targets."""

  def __init__(self, cfg: JointEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_joint_effort_target(
      self._processed_actions, joint_ids=self._target_ids
    )


##
# Tendon actions.
##


@dataclass(kw_only=True)
class TendonLengthActionCfg(BaseActionCfg):
  """Configuration for tendon length control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.TENDON

  def build(self, env: ManagerBasedRlEnv) -> TendonLengthAction:
    return TendonLengthAction(self, env)


@dataclass(kw_only=True)
class TendonVelocityActionCfg(BaseActionCfg):
  """Configuration for tendon velocity control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.TENDON

  def build(self, env: ManagerBasedRlEnv) -> TendonVelocityAction:
    return TendonVelocityAction(self, env)


@dataclass(kw_only=True)
class TendonEffortActionCfg(BaseActionCfg):
  """Configuration for tendon effort control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.TENDON

  def build(self, env: ManagerBasedRlEnv) -> TendonEffortAction:
    return TendonEffortAction(self, env)


class TendonLengthAction(BaseAction):
  """Control tendons via length targets."""

  def __init__(self, cfg: TendonLengthActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_tendon_len_target(
      self._processed_actions, tendon_ids=self._target_ids
    )


class TendonVelocityAction(BaseAction):
  """Control tendons via velocity targets."""

  def __init__(self, cfg: TendonVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_tendon_vel_target(
      self._processed_actions, tendon_ids=self._target_ids
    )


class TendonEffortAction(BaseAction):
  """Control tendons via effort targets."""

  def __init__(self, cfg: TendonEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_tendon_effort_target(
      self._processed_actions, tendon_ids=self._target_ids
    )


##
# Site actions.
##


@dataclass(kw_only=True)
class SiteEffortActionCfg(BaseActionCfg):
  """Configuration for site effort control."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.SITE

  def build(self, env: ManagerBasedRlEnv) -> SiteEffortAction:
    return SiteEffortAction(self, env)


class SiteEffortAction(BaseAction):
  """Control sites via effort targets."""

  def __init__(self, cfg: SiteEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._entity.set_site_effort_target(
      self._processed_actions, site_ids=self._target_ids
    )
