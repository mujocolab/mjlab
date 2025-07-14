from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
  print("hi")


def reset_root_state_uniform(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
):
  pass


def reset_joints_by_scale(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
):
  pass
  