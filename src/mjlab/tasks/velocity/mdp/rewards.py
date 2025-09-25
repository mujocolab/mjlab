from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def subtree_angmom_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_name: str = "robot",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_name]
  if sensor_name not in asset.sensor_names:
    raise ValueError(
      f"Sensor '{sensor_name}' not found in asset '{asset_name}'. "
      f"Available sensors: {asset.sensor_names}"
    )
  angmom_w = asset.data.sensor_data[sensor_name]
  angmom_xy_w = angmom_w[:, :2]
  return torch.sum(torch.square(angmom_xy_w), dim=1)
