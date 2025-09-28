from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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


class feet_air_time:
  """Reward long steps taken by the feet.

  This rewards the agent for lifting feet off the ground for longer than a threshold.
  Provides continuous reward signal during flight phase and smooth command scaling.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.threshold_min = cfg.params["threshold_min"]
    self.threshold_max = cfg.params.get("threshold_max", self.threshold_min + 0.3)
    self.asset_name = cfg.params["asset_name"]
    self.sensor_names = cfg.params["sensor_names"]
    self.num_feet = len(self.sensor_names)
    self.command_name = cfg.params["command_name"]
    self.command_threshold = cfg.params["command_threshold"]
    self.reward_mode = cfg.params.get("reward_mode", "continuous")
    self.command_scale_type = cfg.params.get("command_scale_type", "smooth")
    self.command_scale_width = cfg.params.get("command_scale_width", 0.2)

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

    in_contact = torch.stack(contact_list, dim=1)
    in_air = ~in_contact

    # Detect first contact (landing).
    first_contact = (self.current_air_time > 0) & in_contact

    # Save air time when landing.
    self.last_air_time = torch.where(
      first_contact, self.current_air_time, self.last_air_time
    )

    # Update air time and contact time.
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

    if self.reward_mode == "continuous":
      # Give constant reward of 1.0 for each foot that's in air and above threshold.
      exceeds_min = self.current_air_time > self.threshold_min
      below_max = self.current_air_time <= self.threshold_max
      reward_per_foot = torch.where(
        in_air & exceeds_min & below_max,
        torch.ones_like(self.current_air_time),
        torch.zeros_like(self.current_air_time),
      )
      reward = torch.sum(reward_per_foot, dim=1)
    else:
      # This mode gives (air_time - threshold) as reward on landing.
      air_time_over_min = (self.last_air_time - self.threshold_min).clamp(min=0.0)
      air_time_clamped = air_time_over_min.clamp(
        max=self.threshold_max - self.threshold_min
      )
      reward = torch.sum(air_time_clamped * first_contact, dim=1) / env.step_dt

    command = env.command_manager.get_command(self.command_name)
    assert command is not None
    command_norm = torch.norm(command[:, :2], dim=1)
    if self.command_scale_type == "smooth":
      scale = 0.5 * (
        1.0
        + torch.tanh((command_norm - self.command_threshold) / self.command_scale_width)
      )
      reward *= scale
    else:
      reward *= command_norm > self.command_threshold
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.current_air_time[env_ids] = 0.0
    self.current_contact_time[env_ids] = 0.0
    self.last_air_time[env_ids] = 0.0


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
  sensor_names: list[str],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  contact_list = []
  for sensor_name in sensor_names:
    sensor_data = asset.data.sensor_data[sensor_name]
    foot_contact = sensor_data[:, 0] > 0
    contact_list.append(foot_contact)
  contacts = torch.stack(contact_list, dim=1)
  geom_vel = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2]
  return torch.sum(geom_vel.norm(dim=-1) * contacts, dim=1)
