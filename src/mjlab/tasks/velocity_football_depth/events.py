"""Camera randomization events for depth-football deployment robustness."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def _batched_quat_slerp(
  first: torch.Tensor,
  second: torch.Tensor,
  blend: torch.Tensor,
) -> torch.Tensor:
  """Interpolate WXYZ quaternions with one blend value per environment."""
  dot = torch.sum(first * second, dim=-1, keepdim=True)
  second = torch.where(dot < 0.0, -second, second)
  dot = torch.abs(dot).clamp(max=1.0)
  angle = torch.acos(dot)
  sin_angle = torch.sin(angle)
  linear = (1.0 - blend) * first + blend * second
  spherical = (
    torch.sin((1.0 - blend) * angle) / sin_angle.clamp_min(1.0e-7) * first
    + torch.sin(blend * angle) / sin_angle.clamp_min(1.0e-7) * second
  )
  result = torch.where(sin_angle < 1.0e-6, linear, spherical)
  return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(
    1.0e-7
  )


@requires_model_fields("cam_pos", "cam_quat")
def randomize_camera_between_calibrations(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  first_position: tuple[float, float, float],
  first_quaternion: tuple[float, float, float, float],
  second_position: tuple[float, float, float],
  second_quaternion: tuple[float, float, float, float],
  position_residual_range: tuple[float, float] = (-0.005, 0.005),
  rotation_residual_range: tuple[float, float] = (-0.05236, 0.05236),
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
  """Sample plausible extrinsics along two measured calibration solutions.

  Sampling the calibration interpolation avoids the unrealistic corner rotations
  produced by a wide independent Euler-angle box while still covering both known
  mounting estimates. A small translation/RPY residual covers measurement and
  bracket repeatability error around that calibration manifold.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.int)

  robot = env.scene[asset_cfg.name]
  camera_ids = robot.indexing.cam_ids[asset_cfg.camera_ids]
  num_envs = env_ids.numel()
  num_cameras = camera_ids.numel()
  blend = torch.rand(num_envs, 1, 1, device=env.device)

  first_pos = torch.tensor(first_position, device=env.device).view(1, 1, 3)
  second_pos = torch.tensor(second_position, device=env.device).view(1, 1, 3)
  position = (1.0 - blend) * first_pos + blend * second_pos
  position = position.expand(num_envs, num_cameras, 3).clone()
  position += torch.empty_like(position).uniform_(*position_residual_range)

  first_quat = torch.tensor(first_quaternion, device=env.device).view(1, 1, 4)
  second_quat = torch.tensor(second_quaternion, device=env.device).view(1, 1, 4)
  first_quat = first_quat.expand(num_envs, num_cameras, 4)
  second_quat = second_quat.expand(num_envs, num_cameras, 4)
  quaternion = _batched_quat_slerp(first_quat, second_quat, blend)

  residual = torch.empty(num_envs, num_cameras, 3, device=env.device).uniform_(
    *rotation_residual_range
  )
  residual_quat = quat_from_euler_xyz(
    residual[..., 0].flatten(),
    residual[..., 1].flatten(),
    residual[..., 2].flatten(),
  ).reshape(num_envs, num_cameras, 4)
  quaternion = quat_mul(residual_quat, quaternion)

  env_grid, camera_grid = torch.meshgrid(env_ids, camera_ids, indexing="ij")
  env.sim.model.cam_pos[env_grid, camera_grid] = position
  env.sim.model.cam_quat[env_grid, camera_grid] = quaternion


@requires_model_fields("cam_pos", "cam_quat")
def randomize_camera_between_uncertain_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  lower_position: tuple[float, float, float],
  lower_quaternion: tuple[float, float, float, float],
  upper_position: tuple[float, float, float],
  upper_quaternion: tuple[float, float, float, float],
  alpha_range: tuple[float, float] = (0.0, 2.0 / 3.0),
  fixed_lateral_position: float = 0.01753,
  lower_x_residual_range: tuple[float, float] = (-0.03, 0.03),
  lower_z_residual_range: tuple[float, float] = (-0.01, 0.01),
  lower_pitch_residual_range: tuple[float, float] = (-0.05236, 0.05236),
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
  """Interpolate a one-DOF camera mount from uncertain lower to known upper.

  ``alpha=0`` is the rough real-camera lower limit and ``alpha=1`` the
  official-URDF upper limit. Only the configured prefix of that range is used,
  because the upper end removes the feet from the depth image. The lower-limit
  uncertainty shrinks naturally as alpha approaches the known upper endpoint.
  """
  if not 0.0 <= alpha_range[0] <= alpha_range[1] <= 1.0:
    raise ValueError("alpha_range must be ordered and contained in [0, 1]")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.int)

  robot = env.scene[asset_cfg.name]
  camera_ids = robot.indexing.cam_ids[asset_cfg.camera_ids]
  num_envs = env_ids.numel()
  num_cameras = camera_ids.numel()
  shape = (num_envs, num_cameras)
  alpha = torch.empty(num_envs, 1, 1, device=env.device).uniform_(*alpha_range)

  lower_pos = torch.tensor(lower_position, device=env.device).view(1, 1, 3)
  lower_pos = lower_pos.expand(num_envs, num_cameras, 3).clone()
  lower_pos[..., 0] += torch.empty(shape, device=env.device).uniform_(
    *lower_x_residual_range
  )
  lower_pos[..., 1] = fixed_lateral_position
  lower_pos[..., 2] += torch.empty(shape, device=env.device).uniform_(
    *lower_z_residual_range
  )

  upper_pos = torch.tensor(upper_position, device=env.device).view(1, 1, 3)
  upper_pos = upper_pos.expand(num_envs, num_cameras, 3).clone()
  upper_pos[..., 1] = fixed_lateral_position
  position = (1.0 - alpha) * lower_pos + alpha * upper_pos

  lower_quat = torch.tensor(lower_quaternion, device=env.device).view(1, 1, 4)
  lower_quat = lower_quat.expand(num_envs, num_cameras, 4)
  pitch = torch.empty(shape, device=env.device).uniform_(
    *lower_pitch_residual_range
  )
  zeros = torch.zeros_like(pitch)
  pitch_quat = quat_from_euler_xyz(
    zeros.flatten(), pitch.flatten(), zeros.flatten()
  ).reshape(num_envs, num_cameras, 4)
  lower_quat = quat_mul(pitch_quat, lower_quat)

  upper_quat = torch.tensor(upper_quaternion, device=env.device).view(1, 1, 4)
  upper_quat = upper_quat.expand(num_envs, num_cameras, 4)
  quaternion = _batched_quat_slerp(lower_quat, upper_quat, alpha)

  env_grid, camera_grid = torch.meshgrid(env_ids, camera_ids, indexing="ij")
  env.sim.model.cam_pos[env_grid, camera_grid] = position
  env.sim.model.cam_quat[env_grid, camera_grid] = quaternion
