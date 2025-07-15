"""Useful methods for MDP observations."""

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv
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
  env: ManagerBasedRLEnv,
  entity_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  return env.sim.data.qpos[:, entity_cfg.qpos_ids]
