from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def subtree_angmom_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_name: str = "robot",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_name]
  if sensor_name not in asset.sensor_names:
    raise ValueError(
      f"Sensor '{sensor_name}' not found in asset '{asset_name}'. "
      f"Available sensors: {asset.sensor_names}"
    )
  angmom_w = asset.data.sensor_data[sensor_name]
  angmom_xy_w = angmom_w[:, :2]
  return torch.sum(torch.square(angmom_xy_w), dim=1)


class feet_air_time:
  """Reward long steps taken by the feet.

  This rewards the agent for lifting feet off the ground for longer than a threshold.
  Tracks air time internally and rewards when foot makes contact after being in air.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.threshold = cfg.params["threshold"]
    self.asset_name = cfg.params["asset_name"]
    self.sensor_names = cfg.params["sensor_names"]
    self.command_name = cfg.params["command_name"]
    self.command_threshold = cfg.params["command_threshold"]
    self.num_feet = len(self.sensor_names)

    asset: Entity = env.scene[self.asset_name]
    for sensor_name in self.sensor_names:
      if sensor_name not in asset.sensor_names:
        raise ValueError(
          f"Sensor '{sensor_name}' not found in asset '{self.asset_name}'"
        )

    self.current_air_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)
    self.current_contact_time = torch.zeros(
      env.num_envs, self.num_feet, device=env.device
    )
    self.last_air_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    asset: Entity = env.scene[self.asset_name]

    contact_list = []
    for sensor_name in self.sensor_names:
      sensor_data = asset.data.sensor_data[sensor_name]
      foot_contact = sensor_data[:, 0] > 0
      contact_list.append(foot_contact)

    in_contact = torch.stack(contact_list, dim=1)  # (num_envs, num_feet)

    # Detect first contact.
    first_contact = (self.current_air_time > 0) & in_contact

    # Save air time when landing.
    self.last_air_time = torch.where(
      first_contact, self.current_air_time, self.last_air_time
    )

    self.current_air_time = torch.where(
      in_contact,
      torch.zeros_like(self.current_air_time),  # Reset when in contact.
      self.current_air_time + env.step_dt,  # Increment when in air.
    )

    self.current_contact_time = torch.where(
      in_contact,
      self.current_contact_time + env.step_dt,  # Increment when in contact.
      torch.zeros_like(self.current_contact_time),  # Reset when in air.
    )

    air_time_over_threshold = (self.last_air_time - self.threshold).clamp(min=0.0)
    reward = torch.sum(air_time_over_threshold * first_contact, dim=1)

    # Only reward if command is above threshold.
    command = env.command_manager.get_command(self.command_name)
    assert command is not None
    command_norm = torch.norm(command[:, :2], dim=1)
    reward *= command_norm > self.command_threshold

    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.current_air_time[env_ids] = 0.0
    self.current_contact_time[env_ids] = 0.0
    self.last_air_time[env_ids] = 0.0
