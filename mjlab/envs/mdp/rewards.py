"""Useful methods for MPD rewards."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entities.robots.robot import Robot
from mjlab.sensors import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
  asset: Robot = env.scene[asset_cfg.name]
  lin_vel_error = torch.sum(
    torch.square(
      env.command_manager.get_command(command_name)[:, :2]
      - asset.data.root_link_lin_vel_b[:, :2]
    ),
    dim=1,
  )
  return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
  asset: Robot = env.scene[asset_cfg.name]
  ang_vel_error = torch.square(
    env.command_manager.get_command(command_name)[:, 2]
    - asset.data.root_link_ang_vel_b[:, 2]
  )
  return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize z-axis base linear velocity using L2 squared kernel."""
  asset: Robot = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize xy-axis base angular velocity using L2 squared kernel."""
  asset: Robot = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize non-flat base orientation using L2 squared kernel.

  This is computed by penalizing the xy-components of the projected gravity vector.
  """
  asset: Robot = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_torques_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize joint torques applied on the articulation using L2 squared kernel.

  NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
  """
  asset: Robot = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.data.actuator_force), dim=1)


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel.

  NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
  """
  asset: Robot = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize joint positions if they cross the soft limits.

  This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
  """
  asset: Robot = env.scene[asset_cfg.name]
  out_of_limits = -(
    asset.data.joint_pos - asset.data.soft_joint_pos_limits[:, :, 0]
  ).clip(max=0.0)
  out_of_limits += (
    asset.data.joint_pos - asset.data.soft_joint_pos_limits[:, :, 1]
  ).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


def upright(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset: Robot = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  desired_projected_gravity = torch.tensor([0, 0, -1], device=projected_gravity.device)
  error = torch.sum(torch.square(projected_gravity - desired_projected_gravity), dim=1)
  return torch.exp(-2.0 * error)


def posture(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset: Robot = env.scene.entities[asset_cfg.name]
  error = torch.sum(
    torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
  )
  return torch.exp(-0.5 * error)


def undesired_contacts(
  env: ManagerBasedRlEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize undesired contacts as the number of violations that are above a threshold."""
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  net_contact_forces = contact_sensor.data.net_forces_w_history
  is_contact = (
    torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[
      0
    ]
    > threshold
  )
  return torch.sum(is_contact, dim=1)
