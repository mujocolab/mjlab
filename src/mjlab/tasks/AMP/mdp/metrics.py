from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionCommand


def compute_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Compute Mean Per-Keybody Position Error (MPKPE).

  MPKPE measures the average Euclidean distance between the reference and
  actual positions of all key bodies in world frame.
  """
  pos_error = command.body_pos_relative_w - command.robot_body_pos_w
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def compute_root_relative_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Compute Root-relative Mean Per-Keybody Position Error (R-MPKPE).

  R-MPKPE measures pose error independent of global drift by computing
  positions relative to the root/anchor body.
  """
  # Compute reference positions relative to reference anchor.
  ref_anchor_pos = command.anchor_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
  ref_rel_pos = command.body_pos_w - ref_anchor_pos  # (num_envs, num_bodies, 3)

  # Compute robot positions relative to robot anchor.
  robot_anchor_pos = command.robot_anchor_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
  robot_rel_pos = (
    command.robot_body_pos_w - robot_anchor_pos
  )  # (num_envs, num_bodies, 3)

  # Compute error between relative positions.
  pos_error = ref_rel_pos - robot_rel_pos
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def compute_joint_velocity_error(command: MotionCommand) -> torch.Tensor:
  """Compute average joint velocity error."""
  vel_error = command.joint_vel - command.robot_joint_vel
  return torch.norm(vel_error, dim=-1)  # (num_envs,)


def compute_ee_position_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Compute end effector position error."""
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_pos = command.body_pos_relative_w[:, ee_indices]
  robot_ee_pos = command.robot_body_pos_w[:, ee_indices]

  pos_error = ref_ee_pos - robot_ee_pos
  per_ee_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def compute_ee_orientation_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Compute end effector orientation error."""
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_quat = command.body_quat_relative_w[:, ee_indices]
  robot_ee_quat = command.robot_body_quat_w[:, ee_indices]

  per_ee_error = quat_error_magnitude(ref_ee_quat, robot_ee_quat)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def _get_body_indices(
  command: MotionCommand,
  body_names: tuple[str, ...],
) -> list[int]:
  """Get indices of specified bodies within the command's body list.

  Args:
    command: The motion command.
    body_names: Names of bodies to find.

  Returns:
    List of indices into command.cfg.body_names.
  """
  return [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
