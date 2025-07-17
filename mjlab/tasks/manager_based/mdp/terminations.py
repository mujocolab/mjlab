from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv


def bad_orientation(
  env: ManagerBasedRLEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
  imu_site = asset_cfg.site_ids[0]
  mat = env.sim.data.site_xmat[:, imu_site]  # (batch, 3, 3)
  gravity_world = torch.tensor([0.0, 0.0, 1.0], device=mat.device)
  mat_transposed = mat.transpose(-2, -1)
  projected_gravity = torch.matmul(mat_transposed, gravity_world)
  return torch.any(projected_gravity[:, 2] < threshold, dim=-1)
