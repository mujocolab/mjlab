"""Reusable reward terms over articulation state; weights belong in env cfgs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import body_difference

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def forward_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  if asset.is_fixed_base:
    return asset.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1)
  return asset.data.root_link_lin_vel_w[:, 0]


def com_forward_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  velocity = asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids, 0]
  mass = asset.data.model.body_mass[:, asset.indexing.body_ids[asset_cfg.body_ids]]
  return (velocity * mass).sum(-1) / mass.sum(-1)


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.action.square().sum(-1)


def contact_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  force_range: tuple[float, float] | None = None,
  cost_weight: float = 1.0,
  max_cost: float = float("inf"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  force = asset.data.data.cfrc_ext[:, asset.indexing.body_ids[asset_cfg.body_ids]]
  if force_range is not None:
    force = force.clamp(*force_range)
  return (cost_weight * force.square().sum((1, 2))).clamp(max=max_cost)


def body_distance(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return torch.linalg.vector_norm(body_difference(env, asset_cfg), dim=-1)


def tip_distance(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  position = asset.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return 0.01 * position[:, 0].square() + (position[:, 2] - 2).square()


def standup_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] / env.physics_dt
