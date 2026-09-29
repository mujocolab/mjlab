"""Reset individual articulations and target joints through Entity write APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_gaussian, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def reset_joints_with_normal_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  position_noise: float,
  velocity_noise: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  v = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  q += sample_uniform(-position_noise, position_noise, q.shape, env.device)
  v += sample_gaussian(0.0, velocity_noise, v.shape, env.device)
  limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  q = q.clamp(limits[..., 0], limits[..., 1])
  asset.write_joint_state_to_sim(
    q,
    v,
    env_ids=env_ids,
    joint_ids=(
      torch.tensor(asset_cfg.joint_ids, device=env.device)
      if isinstance(asset_cfg.joint_ids, list)
      else asset_cfg.joint_ids
    ),
  )


def reset_pusher_object(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> None:
  asset: Entity = env.scene[asset_cfg.name]
  n = len(env_ids)
  q = torch.zeros((n, 4), device=env.device)
  pending = torch.ones(n, dtype=torch.bool, device=env.device)
  while pending.any():
    sample = torch.rand((n, 2), device=env.device)
    sample = sample * sample.new_tensor([0.3, 0.4]) + sample.new_tensor([-0.3, -0.2])
    q[pending, :2] = sample[pending]
    pending = q[:, :2].square().sum(-1) <= 0.17**2
  asset.write_joint_state_to_sim(
    q,
    torch.zeros_like(q),
    env_ids=env_ids,
    joint_ids=(
      torch.tensor(asset_cfg.joint_ids, device=env.device)
      if isinstance(asset_cfg.joint_ids, list)
      else asset_cfg.joint_ids
    ),
  )
