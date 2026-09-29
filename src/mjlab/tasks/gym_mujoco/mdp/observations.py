"""Composable benchmark observations through mjlab's articulation interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import joint_pos_rel
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def joint_pos_sin(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  return joint_pos_rel(env, asset_cfg=asset_cfg).sin()


def joint_pos_cos(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  return joint_pos_rel(env, asset_cfg=asset_cfg).cos()


def root_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def root_orientation(
  env: ManagerBasedRlEnv,
  representation: Literal["rotation_6d", "quaternion"] = "rotation_6d",
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.root_link_quat_w
  if representation == "quaternion":
    return quat
  if representation != "rotation_6d":
    raise ValueError(f"Unknown rotation representation: {representation}")
  # First column followed by second column, expressed in the world frame.
  return matrix_from_quat(quat)[..., :, :2].transpose(-2, -1).flatten(1)


def root_lin_vel_w(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w


def body_position(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_link_pos_w[:, asset_cfg.body_ids].flatten(1)


def body_difference(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, dimensions: int = 3
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  positions = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  return (positions[:, 0] - positions[:, 1])[:, :dimensions]


def body_inertia(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.data.cinert[:, asset.indexing.body_ids[asset_cfg.body_ids]].flatten(
    1
  )


def body_spatial_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.data.cvel[:, asset.indexing.body_ids[asset_cfg.body_ids]].flatten(1)


def contact_wrench(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.data.cfrc_ext[
    :, asset.indexing.body_ids[asset_cfg.body_ids]
  ].flatten(1)


def joint_actuator_force(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.qfrc_actuator[:, asset_cfg.joint_ids]


def joint_constraint_force(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.data.qfrc_constraint[:, asset.indexing.joint_v_adr][
    :, asset_cfg.joint_ids
  ]
