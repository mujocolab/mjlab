from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import torch

from .commands import MotionCommand
from .rewards import _get_body_indexes
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entities import Robot


def bad_ref_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command: MotionCommand = env.command_manager.get_term(command_name)
  return torch.norm(command.ref_pos_w - command.robot_ref_pos_w, dim=1) > threshold


def bad_ref_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command: MotionCommand = env.command_manager.get_term(command_name)
  return (
    torch.abs(command.ref_pos_w[:, -1] - command.robot_ref_pos_w[:, -1]) > threshold
  )


def bad_ref_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Robot = env.scene[asset_cfg.name]

  command: MotionCommand = env.command_manager.get_term(command_name)
  motion_projected_gravity_b = quat_apply_inverse(
    command.ref_quat_w, asset.data.GRAVITY_VEC_W
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_ref_quat_w, asset.data.GRAVITY_VEC_W
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: Optional[list[str]] = None,
) -> torch.Tensor:
  command: MotionCommand = env.command_manager.get_term(command_name)

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)
