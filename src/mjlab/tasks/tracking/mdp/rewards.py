from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_body_indexes(
  command: MotionCommand, body_names: Optional[list[str]]
) -> torch.Tensor:
  """Get body indexes as a tensor for efficient indexing."""
  if body_names is None:
    return command._all_body_indexes

  # Check cache first
  cache_key = tuple(sorted(body_names))
  if cache_key not in command._body_indexes_cache:
    indexes = [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
    command._body_indexes_cache[cache_key] = torch.tensor(
      indexes, dtype=torch.long, device=command.device
    )
  return command._body_indexes_cache[cache_key]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: Optional[list[str]] = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  # Optimized: use torch.index_select for cleaner indexing
  error = torch.sum(
    torch.square(
      torch.index_select(command.body_pos_relative_w, 1, body_indexes)
      - torch.index_select(command.robot_body_pos_w, 1, body_indexes)
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: Optional[list[str]] = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  # Optimized: use torch.index_select for cleaner indexing
  error = (
    quat_error_magnitude(
      torch.index_select(command.body_quat_relative_w, 1, body_indexes),
      torch.index_select(command.robot_body_quat_w, 1, body_indexes),
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: Optional[list[str]] = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  # Optimized: use torch.index_select for cleaner indexing
  error = torch.sum(
    torch.square(
      torch.index_select(command.body_lin_vel_w, 1, body_indexes)
      - torch.index_select(command.robot_body_lin_vel_w, 1, body_indexes)
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: Optional[list[str]] = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  # Optimized: use torch.index_select for cleaner indexing
  error = torch.sum(
    torch.square(
      torch.index_select(command.body_ang_vel_w, 1, body_indexes)
      - torch.index_select(command.robot_body_ang_vel_w, 1, body_indexes)
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  asset: Entity = env.scene[asset_cfg.name]
  if sensor_name not in asset.sensor_names:
    raise ValueError(
      f"Sensor '{sensor_name}' not found in asset '{asset_cfg.name}'. "
      f"Available sensors: {asset.sensor_names}"
    )
  # Contact sensor is configured to return max_contacts slots, where the size of each
  # slot is determined by the `data` field in the sensor config.
  # With data='found' and reduce='netforce', only the first slot will be positive
  # if there is any contact and the value will be the number of contacts detected, up
  # to max_contacts.
  # See https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor-contact for
  # more details.
  contact_data = asset.data.sensor_data[sensor_name]  # (num_envs, max_contacts x 1)
  return contact_data[..., 0]  # (num_envs,)
