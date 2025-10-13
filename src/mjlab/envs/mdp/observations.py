"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def base_lin_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


###
# IMU state.
###

# TODO: Make them class based rewards ?


class imu_orientation:
  """Penalize the deviation of the joint positions from the default positions.

  Note: This is implemented as a class so that we can resolve the standard deviation
  dictionary into a tensor and thereafter use it in the __call__ method.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

    asset: Entity = env.scene[self.asset_cfg.name]

    site_name = cfg.params.get("site_name", "imu")

    idxs, _ = asset.find_sites(site_name, preserve_order=True)
    self.site_id = int(asset.indexing.site_ids[idxs[0]].item())

  def __call__(
    self,
    env: ManagerBasedEnv,
    site_name: str = "imu",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del site_name  # Unused.
    del asset_cfg  # Unused.

    asset: Entity = env.scene[self.asset_cfg.name]

    return asset.data.site_quat_w[:, self.site_id, :]


class imu_ang_vel:
  """Penalize the deviation of the joint positions from the default positions.

  Note: This is implemented as a class so that we can resolve the standard deviation
  dictionary into a tensor and thereafter use it in the __call__ method.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

    asset: Entity = env.scene[self.asset_cfg.name]

    site_name = cfg.params.get("site_name", "imu")

    idxs, _ = asset.find_sites(site_name, preserve_order=True)
    self.site_id = int(asset.indexing.site_ids[idxs[0]].item())

  def __call__(
    self,
    env: ManagerBasedEnv,
    site_name: str = "imu",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del site_name  # Unused.
    del asset_cfg  # Unused.

    asset: Entity = env.scene[self.asset_cfg.name]

    ang_vel_w = asset.data.site_ang_vel_w[:, self.site_id, :]
    quat_w = asset.data.site_quat_w[:, self.site_id, :]

    ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)

    return ang_vel_b


class imu_lin_acc:
  """Penalize the deviation of the joint positions from the default positions.

  Note: This is implemented as a class so that we can resolve the standard deviation
  dictionary into a tensor and thereafter use it in the __call__ method.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

    asset: Entity = env.scene[self.asset_cfg.name]

    site_name = cfg.params.get("site_name", "imu")

    idxs, _ = asset.find_sites(site_name, preserve_order=True)
    self.site_id = int(asset.indexing.site_ids[idxs[0]].item())

  def __call__(
    self,
    env: ManagerBasedEnv,
    site_name: str = "imu",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del site_name  # Unused.
    del asset_cfg  # Unused.

    asset: Entity = env.scene[self.asset_cfg.name]

    lin_acc_w = asset.data.site_lin_acc_w[:, self.site_id, :]
    quat_w = asset.data.site_quat_w[:, self.site_id, :]

    lin_acc_b = quat_apply_inverse(quat_w, lin_acc_w)

    return lin_acc_b


class imu_projected_gravity:
  """Penalize the deviation of the joint positions from the default positions.

  Note: This is implemented as a class so that we can resolve the standard deviation
  dictionary into a tensor and thereafter use it in the __call__ method.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

    asset: Entity = env.scene[self.asset_cfg.name]

    site_name = cfg.params.get("site_name", "imu")

    idxs, _ = asset.find_sites(site_name, preserve_order=True)
    self.site_id = int(asset.indexing.site_ids[idxs[0]].item())

  def __call__(
    self,
    env: ManagerBasedEnv,
    site_name: str = "imu",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del site_name  # Unused.
    del asset_cfg  # Unused.

    asset: Entity = env.scene[self.asset_cfg.name]

    quat_w = asset.data.site_quat_w[:, self.site_id, :]

    projected_gravity_b = quat_apply_inverse(quat_w, asset.data.gravity_vec_w)

    return projected_gravity_b


##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


##
# Actions.
##


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command
