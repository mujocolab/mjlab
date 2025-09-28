"""Useful methods for MPD terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when the asset's orientation exceeds the limit angle."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle


def root_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the asset's root height is below the minimum height."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < minimum_height
