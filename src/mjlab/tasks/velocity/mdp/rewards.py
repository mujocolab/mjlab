from __future__ import annotations

from typing import TYPE_CHECKING

import torch

# from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def subtree_angmom_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  # asset: Entity = env.scene[asset_cfg.name]
  # angmom_w = asset.data.data.sensordata[asset_cfg.sensor_ids]
  raise NotImplementedError
