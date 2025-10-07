"""Useful methods for MDP rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
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
  return torch.sum(torch.square(asset.data.actuator_force), dim=1)


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


class joint_pos_limits:
  """Penalize joint positions if they cross the soft limits.

  Implemented as a class to pre-compute joint indexing and avoid runtime overhead.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset: Entity = env.scene[asset_cfg.name]

    # Pre-compute whether we're using all joints or a subset
    self.use_all_joints = isinstance(
      asset_cfg.joint_ids, slice
    ) and asset_cfg.joint_ids == slice(None)

    if not self.use_all_joints:
      # Convert joint_ids to tensor for efficient indexing
      if isinstance(asset_cfg.joint_ids, slice):
        indices = list(range(*asset_cfg.joint_ids.indices(asset.num_joints)))
      else:
        indices = asset_cfg.joint_ids
      self.joint_ids = torch.tensor(indices, dtype=torch.long, device=env.device)

    self.asset_name = asset_cfg.name

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[self.asset_name]
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None

    # Use pre-computed indexing decision
    if self.use_all_joints:
      joint_pos = asset.data.joint_pos
      soft_limits = soft_joint_pos_limits
    else:
      joint_pos = torch.index_select(asset.data.joint_pos, 1, self.joint_ids)
      soft_limits = torch.index_select(soft_joint_pos_limits, 1, self.joint_ids)

    # Compute violations
    out_of_limits = -(joint_pos - soft_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_limits[:, :, 1]).clip(min=0.0)
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

    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )

    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(
    self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    del std  # Unused.
    asset: Entity = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


def electrical_power_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize electrical power consumption of actuators."""
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force
  qd = asset.data.joint_vel
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
