"""Useful methods for MDP observations."""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

##
# Root state.
##


def root_pos(env, entity_cfg: SceneEntityCfg) -> torch.Tensor:
  # env.scene[entity_cfg.name]
  pass


def root_quat(env, entity_cfg: SceneEntityCfg) -> torch.Tensor:
  pass


##
# Joint state.
##


def joint_pos(
  env, entity_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  arr = env.data.qpos[entity_cfg.qpos_ids]
  return torch.from_numpy(arr)
