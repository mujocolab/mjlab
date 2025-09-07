from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensors import ContactSensor

if TYPE_CHECKING:
  from mjlab.entity import Robot
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def feet_air_time(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_cfg: SceneEntityCfg,
  threshold: float,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
  first_contact = contact_sensor.compute_first_contact(env.step_dt)[
    :, sensor_cfg.body_ids
  ]
  assert contact_sensor.data.last_air_time is not None
  last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
  reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
  reward *= (
    torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
  )
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  max_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Robot = env.scene[asset_cfg.name]
  feet_vel = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids]  # (num_envs, 4, 3)
  vel_norm = torch.sqrt(
    torch.linalg.norm(feet_vel[..., :2], dim=-1)
  )  # (num_envs, 4, 1)
  foot_z = asset.data.geom_pos_w[:, asset_cfg.geom_ids][..., -1]  # (num_envs, 4)
  delta = torch.abs(foot_z - max_height)  # (num_envs, 4)
  return torch.sum(delta * vel_norm, dim=-1)
