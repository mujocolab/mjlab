from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  """Periodic gait phase, suppressed while the velocity command is near zero."""
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase_obs = torch.zeros(env.num_envs, 2, device=env.device)
  phase_obs[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase_obs[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  command = env.command_manager.get_command(command_name)
  stand_mask = torch.linalg.norm(command, dim=1) < 0.1
  return torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase_obs), phase_obs)


def ball_pos_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Football position relative to the robot root, expressed in its frame."""
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[asset_cfg.name]
  ball_pos_relative_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  return quat_apply_inverse(robot.data.root_link_quat_w, ball_pos_relative_w)


def ball_vel_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Football velocity relative to the robot root, expressed in its frame."""
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[asset_cfg.name]
  ball_vel_relative_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
  return quat_apply_inverse(robot.data.root_link_quat_w, ball_vel_relative_w)


def ball_to_feet_vectors_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Vectors from the football to selected feet, expressed in the robot frame."""
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[asset_cfg.name]
  feet_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids]
  ball_to_feet_w = feet_pos_w - ball.data.root_link_pos_w[:, None, :]
  robot_quat_w = robot.data.root_link_quat_w[:, None, :].expand(
    -1, ball_to_feet_w.shape[1], -1
  )
  ball_to_feet_b = quat_apply_inverse(robot_quat_w, ball_to_feet_w)
  return ball_to_feet_b.flatten(start_dim=1)


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
