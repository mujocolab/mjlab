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
  sensor_slice = asset.indexing.sensor_adr[sensor_name]
  angmom_w = env.sim.data.sensordata[:, sensor_slice]
  angmom_xy_w = angmom_w[:, :2]
  return torch.sum(torch.square(angmom_xy_w), dim=1)
