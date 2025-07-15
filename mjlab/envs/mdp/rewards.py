"""Useful methods for MPD rewards."""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv


def joint_torques_l2(
  env: ManagerBasedRLEnv,
  entity_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  torques = env.sim.data.qfrc_actuator[:, entity_cfg.dof_ids]
  return torch.sum(torch.square(torques), dim=-1)
