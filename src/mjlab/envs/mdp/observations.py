"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_box_minus

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def base_lin_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Compute relative joint positions (current - default) in DOF space.

  For hinge/slide joints: returns scalar difference.
  For ball joints: returns 3D axis-angle difference via quat_box_minus.

  Returns:
    Tensor of shape (num_envs, total_dof) for selected joints.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos

  # Hinge-only entities (nq == nv means no ball joints).
  if asset.nq == asset.nv:
    q_indices = asset.indexing.expand_to_q_indices(asset_cfg.joint_ids)
    return joint_pos[:, q_indices] - default_joint_pos[:, q_indices]

  # Ball joints.
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    joint_ids_tensor = torch.arange(asset.num_joints, device=joint_pos.device)
  elif isinstance(joint_ids, list):
    joint_ids_tensor = torch.tensor(joint_ids, device=joint_pos.device)
  else:
    joint_ids_tensor = joint_ids

  qpos_widths = asset.indexing.joint_qpos_widths[joint_ids_tensor]
  dof_widths = asset.indexing.joint_dof_widths[joint_ids_tensor]
  q_offsets = asset.indexing.q_offsets[joint_ids_tensor]

  num_envs = joint_pos.shape[0]
  device = joint_pos.device

  is_ball_joint = (qpos_widths == 4) & (dof_widths == 3)
  dof_cumsum = torch.cat(
    [torch.zeros(1, device=device, dtype=dof_widths.dtype), dof_widths.cumsum(0)[:-1]]
  )
  total_dof = int(dof_widths.sum().item())
  result = torch.zeros((num_envs, total_dof), device=device, dtype=joint_pos.dtype)

  # Ball joints: quaternion difference via quat_box_minus.
  if is_ball_joint.any():
    ball_q_offsets = q_offsets[is_ball_joint]
    ball_out_offsets = dof_cumsum[is_ball_joint]

    ball_q_indices = ball_q_offsets.unsqueeze(1) + torch.arange(4, device=device)
    current_quats = joint_pos[:, ball_q_indices.flatten()].view(num_envs, -1, 4)
    default_quats = default_joint_pos[:, ball_q_indices.flatten()].view(num_envs, -1, 4)

    num_ball = current_quats.shape[1]
    diff_flat = quat_box_minus(
      current_quats.reshape(-1, 4), default_quats.reshape(-1, 4)
    )
    diff = diff_flat.view(num_envs, num_ball, 3)

    ball_out_indices = ball_out_offsets.unsqueeze(1) + torch.arange(3, device=device)
    result[:, ball_out_indices.flatten()] = diff.reshape(num_envs, -1)

  # Hinge joints: scalar subtraction.
  is_hinge_joint = ~is_ball_joint
  if is_hinge_joint.any():
    hinge_q_offsets = q_offsets[is_hinge_joint]
    hinge_out_offsets = dof_cumsum[is_hinge_joint]
    diff_hinge = joint_pos[:, hinge_q_offsets] - default_joint_pos[:, hinge_q_offsets]
    result[:, hinge_out_offsets] = diff_hinge

  return result


def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  v_indices = asset.indexing.expand_to_v_indices(asset_cfg.joint_ids)
  return asset.data.joint_vel[:, v_indices] - default_joint_vel[:, v_indices]


##
# Actions.
##


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


##
# Sensors.
##


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Get observation from a built-in sensor by name."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  offset: float = 0.0,
  miss_value: float | None = None,
) -> torch.Tensor:
  """Height scan from a raycast sensor.

  Returns the height of the sensor frame above each hit point.

  Args:
    env: The environment.
    sensor_name: Name of a RayCastSensor in the scene.
    offset: Constant offset subtracted from heights.
    miss_value: Value to use for rays that miss (distance < 0).
      Defaults to the sensor's ``max_distance``.

  Returns:
    Tensor of shape [B, N] where B is num_envs and N is num_rays.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  if miss_value is None:
    miss_value = sensor.cfg.max_distance
  heights = (
    sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.hit_pos_w[..., 2] - offset
  )
  miss_mask = sensor.data.distances < 0
  return torch.where(miss_mask, torch.full_like(heights, miss_value), heights)
