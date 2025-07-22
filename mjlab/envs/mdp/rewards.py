"""Useful methods for MPD rewards."""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv
from mjlab.entities.robots.robot import Robot


def track_lin_vel_xy_exp(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
  asset = env.scene.entities[asset_cfg.name]
  lin_vel_error = torch.sum(
    torch.square(
      env.command_manager.get_command(command_name)[:, :2]
      - asset.data.root_com_lin_vel_b[:, :2]
    ),
    dim=1,
  )
  return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
  env: ManagerBasedRLEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
  # extract the used quantities (to enable type-hinting)
  asset = env.scene.entities[asset_cfg.name]
  # compute the error
  ang_vel_error = torch.square(
    env.command_manager.get_command(command_name)[:, 2]
    - asset.data.root_com_ang_vel_b[:, 2]
  )
  return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize z-axis base linear velocity using L2 squared kernel."""
  # extract the used quantities (to enable type-hinting)
  asset = env.scene.entities[asset_cfg.name]
  return torch.square(asset.data.root_com_lin_vel_b[:, 2])


def ang_vel_xy_l2(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize xy-axis base angular velocity using L2 squared kernel."""
  # extract the used quantities (to enable type-hinting)
  asset = env.scene.entities[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_com_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize non-flat base orientation using L2 squared kernel.

  This is computed by penalizing the xy-components of the projected gravity vector.
  """
  # extract the used quantities (to enable type-hinting)
  asset = env.scene.entities[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_torques_l2(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize joint torques applied on the articulation using L2 squared kernel.

  NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
  """
  # extract the used quantities (to enable type-hinting)
  asset = env.scene.entities[asset_cfg.name]
  return torch.sum(torch.square(asset.data.data.actuator_force), dim=1)


def joint_acc_l2(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel.

  NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
  """
  # extract the used quantities (to enable type-hinting)
  asset = env.scene.entities[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def joint_pos_limits(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """Penalize joint positions if they cross the soft limits.

  This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
  """
  # extract the used quantities (to enable type-hinting)
  asset = env.scene.entities[asset_cfg.name]
  # compute out of limits constraints
  out_of_limits = -(
    asset.data.joint_pos - asset.data.soft_joint_pos_limits[:, :, 0]
  ).clip(max=0.0)
  out_of_limits += (
    asset.data.joint_pos - asset.data.soft_joint_pos_limits[:, :, 1]
  ).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


def upright(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset = env.scene.entities[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  desired_projected_gravity = torch.tensor([0, 0, -1], device=projected_gravity.device)
  error = torch.sum(torch.square(projected_gravity - desired_projected_gravity), dim=1)
  return torch.exp(-2.0 * error)


def posture(
  env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset: Robot = env.scene.entities[asset_cfg.name]
  error = torch.sum(
    torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
  )
  return torch.exp(-0.5 * error)


# def torso_desired_height(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#   asset: Robot = env.scene.entities[asset_cfg.name]
#   torso_height = asset.data.root_link_pose_w[:, 2]
