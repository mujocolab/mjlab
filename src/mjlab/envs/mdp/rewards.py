"""Useful methods for MDP rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_box_minus
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for being alive."""
  return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize terminated episodes that don't correspond to episodic timeouts."""
  return env.termination_manager.terminated.float()


def joint_torques_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint torques applied on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.square(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1
  )


def joint_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint velocities on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  v_indices = asset.indexing.expand_to_v_indices(asset_cfg.joint_ids)
  return torch.sum(torch.square(asset.data.joint_vel[:, v_indices]), dim=1)


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  v_indices = asset.indexing.expand_to_v_indices(asset_cfg.joint_ids)
  return torch.sum(torch.square(asset.data.joint_acc[:, v_indices]), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel.

  Operates on raw policy output (before per-term scale/offset).
  """
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def action_acc_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the acceleration of the actions using L2 squared kernel.

  Operates on raw policy output (before per-term scale/offset).
  """
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.sum(torch.square(action_acc), dim=1)


def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint positions if they cross the soft limits."""
  asset: Entity = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  q_indices = asset.indexing.expand_to_q_indices(asset_cfg.joint_ids)
  joint_pos = asset.data.joint_pos[:, q_indices]

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    joint_ids_tensor = torch.arange(asset.num_joints, device=env.device)
  elif isinstance(joint_ids, list):
    joint_ids_tensor = torch.tensor(joint_ids, device=env.device)
  else:
    joint_ids_tensor = joint_ids

  # Expand limits to qpos dimensions for ball joints.
  limits = soft_joint_pos_limits[:, joint_ids_tensor]
  qpos_widths = asset.indexing.joint_qpos_widths[joint_ids_tensor]
  expanded_limits = limits.repeat_interleave(qpos_widths, dim=1)

  out_of_limits = -(joint_pos - expanded_limits[..., 0]).clip(max=0.0)
  out_of_limits += (joint_pos - expanded_limits[..., 1]).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


class posture:
  """Penalize the deviation of the joint positions from the default positions.

  Note: This is implemented as a class so that we can resolve the standard deviation
  dictionary into a tensor and thereafter use it in the __call__ method.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    joint_ids, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

    self._joint_dof_widths = asset.indexing.joint_dof_widths[self._joint_ids]
    self._joint_qpos_widths = asset.indexing.joint_qpos_widths[self._joint_ids]
    self._q_offsets = asset.indexing.q_offsets[self._joint_ids]
    self._is_ball_joint = (self._joint_qpos_widths == 4) & (self._joint_dof_widths == 3)

    # Precompute cumulative DOF offsets for output indexing.
    self._dof_cumsum = torch.cat(
      [
        torch.zeros(1, device=env.device, dtype=self._joint_dof_widths.dtype),
        self._joint_dof_widths.cumsum(0)[:-1],
      ]
    )
    self._std_expanded = self.std.repeat_interleave(self._joint_dof_widths)

  def __call__(
    self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    del std  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    num_envs = joint_pos.shape[0]
    device = joint_pos.device

    total_dof = int(self._joint_dof_widths.sum().item())
    error = torch.zeros((num_envs, total_dof), device=device, dtype=joint_pos.dtype)

    # Ball joints: quaternion difference.
    if self._is_ball_joint.any():
      ball_q_offsets = self._q_offsets[self._is_ball_joint]
      ball_out_offsets = self._dof_cumsum[self._is_ball_joint]

      ball_q_indices = ball_q_offsets.unsqueeze(1) + torch.arange(4, device=device)
      current_quats = joint_pos[:, ball_q_indices.flatten()].view(num_envs, -1, 4)
      default_quats = self.default_joint_pos[:, ball_q_indices.flatten()].view(
        num_envs, -1, 4
      )

      num_ball = current_quats.shape[1]
      diff_flat = quat_box_minus(
        current_quats.reshape(-1, 4), default_quats.reshape(-1, 4)
      )
      diff = diff_flat.view(num_envs, num_ball, 3)

      ball_out_indices = ball_out_offsets.unsqueeze(1) + torch.arange(3, device=device)
      error[:, ball_out_indices.flatten()] = diff.reshape(num_envs, -1)

    # Hinge joints: scalar subtraction.
    is_hinge_joint = ~self._is_ball_joint
    if is_hinge_joint.any():
      hinge_q_offsets = self._q_offsets[is_hinge_joint]
      hinge_out_offsets = self._dof_cumsum[is_hinge_joint]

      diff_hinge = (
        joint_pos[:, hinge_q_offsets] - self.default_joint_pos[:, hinge_q_offsets]
      )
      error[:, hinge_out_offsets] = diff_hinge

    error_squared = torch.square(error)
    return torch.exp(-torch.mean(error_squared / (self._std_expanded**2), dim=1))


class electrical_power_cost:
  """Penalize electrical power consumption of actuators."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    joint_ids, _ = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )
    actuator_ids, _ = asset.find_actuators(
      cfg.params["asset_cfg"].joint_names,
    )
    joint_ids_tensor = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self._v_indices = asset.indexing.expand_to_v_indices(joint_ids_tensor)
    self._actuator_ids = torch.tensor(actuator_ids, device=env.device, dtype=torch.long)

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force[:, self._actuator_ids]
    qd = asset.data.joint_vel[:, self._v_indices]
    mech = tau * qd
    mech_pos = torch.clamp(mech, min=0.0)  # Don't penalize regen.
    return torch.sum(mech_pos, dim=1)


def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
