"""Stock v5 unhealthy-state checks over entity coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def root_unhealthy(
  env: ManagerBasedRlEnv,
  z_range: tuple[float, float],
  check_finite: bool = False,
  inclusive: bool = False,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.root_link_pos_w[:, 2]
  lo, hi = z_range
  healthy = (z >= lo) & (z <= hi) if inclusive else (z > lo) & (z < hi)
  if check_finite:
    for value in (
      asset.data.root_link_pose_w,
      asset.data.root_link_vel_w,
      asset.data.joint_pos,
      asset.data.joint_vel,
    ):
      healthy &= torch.isfinite(value).all(-1)
  return ~healthy


def planar_unhealthy(
  env: ManagerBasedRlEnv,
  z_range: tuple[float, float],
  angle_range: tuple[float, float],
  state_range: tuple[float, float] | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  q, v = asset.data.joint_pos, asset.data.joint_vel
  healthy = (q[:, 1] > z_range[0]) & (q[:, 1] < z_range[1])
  healthy &= (q[:, 2] > angle_range[0]) & (q[:, 2] < angle_range[1])
  if state_range is not None:
    state = torch.cat((q[:, 2:], v), dim=-1)
    healthy &= ((state > state_range[0]) & (state < state_range[1])).all(-1)
  return ~healthy


def pendulum_unhealthy(
  env: ManagerBasedRlEnv, angle_limit: float = 0.2, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  q, v = asset.data.joint_pos, asset.data.joint_vel
  return (
    ~torch.isfinite(q).all(-1)
    | ~torch.isfinite(v).all(-1)
    | (q[:, 1].abs() > angle_limit)
  )


def tip_too_low(
  env: ManagerBasedRlEnv, minimum_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2].squeeze(-1) <= minimum_height
