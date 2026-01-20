from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ee_to_object_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from end effector to object in base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  distance_vec_w = obj_pos_w - ee_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def object_to_goal_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from object to goal in base frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  obj_pos_w = obj.data.root_link_pos_w
  goal_pos_w = command.target_pos
  distance_vec_w = goal_pos_w - obj_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def ee_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """EE linear velocity in EE frame."""
  robot: Entity = env.scene[asset_cfg.name]
  ee_vel_w = robot.data.site_vel_w[:, asset_cfg.site_ids].squeeze(1)  # (B, 6)
  ee_vel_linear_w = ee_vel_w[:, :3]
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  ee_vel_linear_ee = quat_apply(quat_inv(ee_quat_w), ee_vel_linear_w)
  return ee_vel_linear_ee


def target_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Target position in EE frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  target_pos_w = command.target_pos
  target_pos_rel_w = target_pos_w - ee_pos_w
  target_pos_ee = quat_apply(quat_inv(ee_quat_w), target_pos_rel_w)
  return target_pos_ee


def camera_rgb(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """RGB observation in CNN-compatible format (B, C, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  rgb_data = sensor.data.rgb  # (B, H, W, 3)
  assert rgb_data is not None, f"Camera '{sensor_name}' has no RGB data"
  rgb_data = rgb_data.permute(0, 3, 1, 2)  # (B, 3, H, W)
  return rgb_data.float() / 255.0


def camera_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  cutoff_distance: float,
  min_depth: float = 0.01,
) -> torch.Tensor:
  """Depth observation in CNN-compatible format (B, 1, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  depth_data = sensor.data.depth  # (B, H, W, 1)
  assert depth_data is not None, f"Camera '{sensor_name}' has no depth data"
  depth_data = depth_data.permute(0, 3, 1, 2)  # (B, 1, H, W)
  depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
  return torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)
