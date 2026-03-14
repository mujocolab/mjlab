from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_HAND_CFG = SceneEntityCfg("robot", body_names=("palm",))
_DEFAULT_PALM_CENTER_GEOM_EXPR = "palm_collision_.*"
_DEFAULT_JOINT_ASSET_CFG = SceneEntityCfg("robot", joint_names=(".*",))


def _joint_position_command(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  action_name: str = "joint_pos",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  term = env.action_manager.get_term(action_name)

  target_ids = getattr(term, "target_ids", None)
  command_value = getattr(term, "_target", None)
  if command_value is None:
    command_value = getattr(term, "_processed_actions", None)
  if target_ids is None or command_value is None:
    return asset.data.joint_pos_target[:, joint_ids]

  target_ids = target_ids.to(device=env.device, dtype=torch.long)
  commanded_full = asset.data.joint_pos.clone()
  commanded_full[:, target_ids] = command_value
  return commanded_full[:, joint_ids]


def joint_pos_commanded(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  asset_cfg: SceneEntityCfg = _DEFAULT_JOINT_ASSET_CFG,
) -> torch.Tensor:
  return _joint_position_command(env=env, asset_cfg=asset_cfg, action_name=action_name)


def joint_pos_command_error(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  biased: bool = True,
  asset_cfg: SceneEntityCfg = _DEFAULT_JOINT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  commanded = _joint_position_command(env=env, asset_cfg=asset_cfg, action_name=action_name)
  measured = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return commanded - measured[:, joint_ids]


def palm_center_pose_w(
  env: ManagerBasedRlEnv,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = _DEFAULT_PALM_CENTER_GEOM_EXPR,
) -> tuple[torch.Tensor, torch.Tensor]:
  hand: Entity = env.scene[hand_cfg.name]
  palm_pos_w = hand.data.body_link_pos_w[:, hand_cfg.body_ids].squeeze(1)
  palm_quat_w = hand.data.body_link_quat_w[:, hand_cfg.body_ids].squeeze(1)

  cache_key = f"_dex_manip_palm_center_geom_ids::{hand_cfg.name}::{palm_center_geom_expr}"
  palm_geom_ids = getattr(env, cache_key, None)
  if palm_geom_ids is None:
    palm_geom_ids, _ = hand.find_geoms(palm_center_geom_expr, preserve_order=True)
    palm_geom_ids = torch.tensor(palm_geom_ids, dtype=torch.long, device=env.device)
    setattr(env, cache_key, palm_geom_ids)

  if palm_geom_ids.numel() == 0:
    return palm_pos_w, palm_quat_w

  palm_center_w = hand.data.geom_pos_w[:, palm_geom_ids].mean(dim=1)
  return palm_center_w, palm_quat_w


def object_pose_in_palm_frame(
  env: ManagerBasedRlEnv,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = _DEFAULT_PALM_CENTER_GEOM_EXPR,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  palm_center_w, palm_quat_w = palm_center_pose_w(
    env=env,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  q_inv = quat_inv(palm_quat_w)
  obj_pos_w = obj.data.root_link_pos_w
  obj_quat_w = obj.data.root_link_quat_w
  pos_palm = quat_apply(q_inv, obj_pos_w - palm_center_w)
  quat_palm = quat_mul(q_inv, obj_quat_w)
  return torch.cat([pos_palm, quat_palm], dim=-1)


def object_lin_vel_in_palm_frame(
  env: ManagerBasedRlEnv,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = _DEFAULT_PALM_CENTER_GEOM_EXPR,
) -> torch.Tensor:
  hand: Entity = env.scene[hand_cfg.name]
  obj: Entity = env.scene[object_name]
  palm_pos_w = hand.data.body_link_pos_w[:, hand_cfg.body_ids].squeeze(1)
  palm_quat_w = hand.data.body_link_quat_w[:, hand_cfg.body_ids].squeeze(1)
  palm_vel_w = hand.data.body_link_vel_w[:, hand_cfg.body_ids].squeeze(1)
  palm_lin_vel_w = palm_vel_w[:, :3]
  palm_ang_vel_w = palm_vel_w[:, 3:]

  palm_center_w, _ = palm_center_pose_w(
    env=env,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  center_offset_w = palm_center_w - palm_pos_w
  palm_center_lin_vel_w = palm_lin_vel_w + torch.cross(
    palm_ang_vel_w, center_offset_w, dim=-1
  )

  obj_lin_vel_w = obj.data.root_link_lin_vel_w
  rel_lin_vel_w = obj_lin_vel_w - palm_center_lin_vel_w
  return quat_apply(quat_inv(palm_quat_w), rel_lin_vel_w)


def object_ang_vel_in_palm_frame(
  env: ManagerBasedRlEnv,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = _DEFAULT_PALM_CENTER_GEOM_EXPR,
) -> torch.Tensor:
  del palm_center_geom_expr
  hand: Entity = env.scene[hand_cfg.name]
  obj: Entity = env.scene[object_name]
  palm_quat_w = hand.data.body_link_quat_w[:, hand_cfg.body_ids].squeeze(1)
  palm_ang_vel_w = hand.data.body_link_vel_w[:, hand_cfg.body_ids].squeeze(1)[:, 3:]
  obj_ang_vel_w = obj.data.root_link_ang_vel_w
  rel_ang_vel_w = obj_ang_vel_w - palm_ang_vel_w
  return quat_apply(quat_inv(palm_quat_w), rel_ang_vel_w)


def object_size(
  env: ManagerBasedRlEnv,
  object_name: str,
  geom_name: str = "object_geom",
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  cache_key = f"_dex_manip_size_geom_id::{object_name}::{geom_name}"
  geom_world_id = getattr(env, cache_key, None)
  if geom_world_id is None:
    geom_local_ids, _ = obj.find_geoms(geom_name, preserve_order=True)
    if len(geom_local_ids) != 1:
      raise ValueError(
        f"Expected exactly one object geom matching '{geom_name}', got {len(geom_local_ids)}."
      )
    geom_world_id = int(obj.indexing.geom_ids[geom_local_ids[0]].item())
    setattr(env, cache_key, geom_world_id)

  size = env.sim.model.geom_size[:, geom_world_id, 0]
  if torch.any(size <= 0.0):
    rbound = env.sim.model.geom_rbound[:, geom_world_id]
    fallback_size = rbound / (3.0**0.5)
    size = torch.where(size > 0.0, size, fallback_size)
  return size.unsqueeze(-1)


def object_mass(
  env: ManagerBasedRlEnv,
  object_name: str,
  body_name: str = "object",
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  cache_key = f"_dex_manip_mass_body_id::{object_name}::{body_name}"
  body_world_id = getattr(env, cache_key, None)
  if body_world_id is None:
    body_local_ids, _ = obj.find_bodies(body_name, preserve_order=True)
    if len(body_local_ids) != 1:
      raise ValueError(
        f"Expected exactly one object body matching '{body_name}', got {len(body_local_ids)}."
      )
    body_world_id = int(obj.indexing.body_ids[body_local_ids[0]].item())
    setattr(env, cache_key, body_world_id)

  mass = env.sim.model.body_mass[:, body_world_id]
  return mass.unsqueeze(-1)


def object_com_offset_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  body_name: str = "object",
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  cache_key = f"_dex_manip_com_body_id::{object_name}::{body_name}"
  body_world_id = getattr(env, cache_key, None)
  if body_world_id is None:
    body_local_ids, _ = obj.find_bodies(body_name, preserve_order=True)
    if len(body_local_ids) != 1:
      raise ValueError(
        f"Expected exactly one object body matching '{body_name}', got {len(body_local_ids)}."
      )
    body_world_id = int(obj.indexing.body_ids[body_local_ids[0]].item())
    setattr(env, cache_key, body_world_id)

  return env.sim.model.body_ipos[:, body_world_id, :]


def object_friction_coeff(
  env: ManagerBasedRlEnv,
  object_name: str,
  geom_name: str = "object_geom",
  axis: int = 0,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  cache_key = f"_dex_manip_friction_geom_id::{object_name}::{geom_name}"
  geom_world_id = getattr(env, cache_key, None)
  if geom_world_id is None:
    geom_local_ids, _ = obj.find_geoms(geom_name, preserve_order=True)
    if len(geom_local_ids) != 1:
      raise ValueError(
        f"Expected exactly one object geom matching '{geom_name}', got {len(geom_local_ids)}."
      )
    geom_world_id = int(obj.indexing.geom_ids[geom_local_ids[0]].item())
    setattr(env, cache_key, geom_world_id)

  friction = env.sim.model.geom_friction[:, geom_world_id, axis]
  return friction.unsqueeze(-1)
