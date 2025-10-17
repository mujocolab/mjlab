from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  desired = torch.zeros_like(actual)
  desired[:, :2] = command[:, :2]
  lin_vel_error = torch.sum(torch.square(desired - actual), dim=1)
  return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  desired = torch.zeros_like(actual)
  desired[:, 2] = command[:, 2]
  ang_vel_error = torch.sum(torch.square(desired - actual), dim=1)
  return torch.exp(-ang_vel_error / std**2)


def feet_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Returns the current air time from the contact sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.current_air_time


def foot_clearance_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  tanh_mult: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  foot_z_target_error = torch.square(
    asset.data.geom_pos_w[:, asset_cfg.geom_ids, 2] - target_height
  )
  foot_velocity_tanh = torch.tanh(
    tanh_mult * torch.norm(asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2], dim=2)
  )
  reward = foot_z_target_error * foot_velocity_tanh
  return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding when in contact with ground."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  contacts = sensor.count > 0
  geom_vel = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2]
  return torch.sum(geom_vel.norm(dim=-1) * contacts, dim=1)
