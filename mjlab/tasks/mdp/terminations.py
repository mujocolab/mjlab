from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from mjlab.entities.robots.robot import Robot

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv


def bad_orientation(
  env: ManagerBasedRLEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
  asset: Robot = env.scene.entities[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return torch.any(-projected_gravity[:, 2] < threshold, dim=-1)
