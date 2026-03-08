from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _anchor_frame_b(
  command: MotionCommand,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute anchor position and orientation in body frame.

  Caches the result on the command object for the current env step so
  that ``motion_anchor_pos_b`` and ``motion_anchor_ori_b`` (which are
  typically both present in the observation group) share a single
  ``subtract_frame_transforms`` call instead of duplicating work.
  """
  step = command._env.common_step_counter
  cache = getattr(command, "_anchor_frame_b_cache", None)
  if cache is not None and cache[0] == step:
    return cache[1], cache[2]

  pos, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  command._anchor_frame_b_cache = (step, pos, ori)  # type: ignore[attr-defined]
  return pos, ori


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  pos, _ = _anchor_frame_b(command)
  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  _, ori = _anchor_frame_b(command)
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def _robot_body_frame_b(
  command: MotionCommand,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute robot body positions/orientations in body frame.

  Same caching pattern as ``_anchor_frame_b`` -- shared across
  ``robot_body_pos_b`` and ``robot_body_ori_b``.
  """
  step = command._env.common_step_counter
  cache = getattr(command, "_body_frame_b_cache", None)
  if cache is not None and cache[0] == step:
    return cache[1], cache[2]

  num_bodies = len(command.cfg.body_names)
  pos_b, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
    command.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  command._body_frame_b_cache = (step, pos_b, ori_b)  # type: ignore[attr-defined]
  return pos_b, ori_b


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  pos_b, _ = _robot_body_frame_b(command)
  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  _, ori_b = _robot_body_frame_b(command)
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
