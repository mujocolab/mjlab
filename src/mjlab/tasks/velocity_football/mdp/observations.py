from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def _shared_ball_perception_error(
  env: ManagerBasedRlEnv,
  shape: torch.Size,
  bias_range: float,
  frame_noise_range: float,
) -> torch.Tensor:
  """Return one shared ball-position error sample for the current control step."""
  if bias_range < 0.0 or frame_noise_range < 0.0:
    raise ValueError("ball perception error ranges must be non-negative")

  env_state = vars(env)
  bias = env_state.get("_football_shared_ball_pos_bias")
  frame_noise = env_state.get("_football_shared_ball_pos_noise")
  step_marker = env_state.get("_football_shared_ball_pos_step")
  if bias is None or frame_noise is None or step_marker is None or bias.shape != shape:
    bias = torch.empty(shape, device=env.device).uniform_(-bias_range, bias_range)
    frame_noise = torch.zeros(shape, device=env.device)
    step_marker = torch.full(
      (env.num_envs,),
      -1,
      dtype=env.episode_length_buf.dtype,
      device=env.device,
    )
    env_state["_football_shared_ball_pos_bias"] = bias
    env_state["_football_shared_ball_pos_noise"] = frame_noise
    env_state["_football_shared_ball_pos_step"] = step_marker

  new_step = step_marker != env.episode_length_buf
  reset_mask = new_step & (env.episode_length_buf == 0)
  if torch.any(reset_mask):
    bias[reset_mask].uniform_(-bias_range, bias_range)
  if torch.any(new_step):
    frame_noise[new_step].uniform_(-frame_noise_range, frame_noise_range)
    step_marker[new_step] = env.episode_length_buf[new_step]
  return bias + frame_noise


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
  """Football XY position relative to the robot root in its yaw-aligned frame."""
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[asset_cfg.name]
  ball_pos_relative_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  ball_pos_yaw = quat_apply_inverse(
    yaw_quat(robot.data.root_link_quat_w), ball_pos_relative_w
  )
  return ball_pos_yaw[:, :2]


def ball_pos_b_with_fixed_bias(
  env: ManagerBasedRlEnv,
  bias_range: float = 0.10,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Football position with a persistent per-episode perception bias."""
  if bias_range < 0.0:
    raise ValueError(f"bias_range must be non-negative, got {bias_range}")
  position = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
  env_state = vars(env)
  bias = env_state.get("_football_ball_pos_bias")
  if bias is None or bias.shape != position.shape:
    bias = torch.empty_like(position).uniform_(-bias_range, bias_range)
    env_state["_football_ball_pos_bias"] = bias
  reset_mask = env.episode_length_buf == 0
  if torch.any(reset_mask):
    bias[reset_mask] = torch.empty_like(bias[reset_mask]).uniform_(
      -bias_range, bias_range
    )
  return position + bias


def perceived_ball_pos_b(
  env: ManagerBasedRlEnv,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Ball position using a shared episode bias and per-frame perception noise."""
  position = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
  error = _shared_ball_perception_error(
    env,
    position.shape,
    bias_range,
    frame_noise_range,
  )
  return position + error


def ball_vel_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Football velocity relative to the robot root in its yaw-aligned frame."""
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[asset_cfg.name]
  ball_vel_relative_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
  return quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), ball_vel_relative_w)


def ball_to_feet_vectors_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Selected foot-to-football XY vectors in the robot yaw-aligned frame."""
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[asset_cfg.name]
  feet_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids]
  feet_to_ball_w = ball.data.root_link_pos_w[:, None, :] - feet_pos_w
  robot_yaw_quat_w = yaw_quat(robot.data.root_link_quat_w)[:, None, :].expand(
    -1, feet_to_ball_w.shape[1], -1
  )
  feet_to_ball_yaw = quat_apply_inverse(robot_yaw_quat_w, feet_to_ball_w)
  return feet_to_ball_yaw[:, :, :2].flatten(start_dim=1)


def perceived_ball_to_feet_vectors_b(
  env: ManagerBasedRlEnv,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Foot-to-ball vectors derived from the same perceived ball position."""
  vectors = ball_to_feet_vectors_b(
    env,
    ball_cfg=ball_cfg,
    asset_cfg=asset_cfg,
  )
  error = _shared_ball_perception_error(
    env,
    torch.Size((env.num_envs, 2)),
    bias_range,
    frame_noise_range,
  )
  num_feet = vectors.shape[1] // 2
  return vectors + error.repeat(1, num_feet)


def _shared_masked_ball_visual(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float],
  y_range: tuple[float, float],
  dropout_probability: float,
  bias_range: float,
  frame_noise_range: float,
  ball_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return synchronized masked ball position, foot vectors, and visibility.

  The result is cached for the current control step because the three actor
  observation terms, and their history-group copies, are evaluated separately.
  """
  if not 0.0 <= dropout_probability <= 1.0:
    raise ValueError("dropout_probability must be in [0, 1]")
  if x_range[0] > x_range[1] or y_range[0] > y_range[1]:
    raise ValueError("ball visibility ranges must be ordered")

  env_state = vars(env)
  cache = cast(
    dict[str, torch.Tensor] | None,
    env_state.get("_football_masked_ball_visual"),
  )
  valid_cache = (
    isinstance(cache, dict)
    and cache["step"].shape == env.episode_length_buf.shape
    and cache["ball_pos"].shape == (env.num_envs, 2)
  )
  if not valid_cache:
    cache = {
      "step": torch.full_like(env.episode_length_buf, -1),
      "ball_pos": torch.zeros(env.num_envs, 2, device=env.device),
      "feet": torch.zeros(env.num_envs, 4, device=env.device),
      "visible": torch.zeros(env.num_envs, 1, device=env.device),
    }
    env_state["_football_masked_ball_visual"] = cache
  assert cache is not None

  new_step = cache["step"] != env.episode_length_buf
  if torch.any(new_step):
    true_ball_pos = ball_pos_b(env, ball_cfg=ball_cfg, asset_cfg=asset_cfg)
    true_feet = ball_to_feet_vectors_b(
      env,
      ball_cfg=ball_cfg,
      asset_cfg=asset_cfg,
    )
    error = _shared_ball_perception_error(
      env,
      true_ball_pos.shape,
      bias_range,
      frame_noise_range,
    )
    num_feet = true_feet.shape[1] // 2
    perceived_ball_pos = true_ball_pos + error
    perceived_feet = true_feet + error.repeat(1, num_feet)

    in_rectangle = (
      (true_ball_pos[:, 0] >= x_range[0])
      & (true_ball_pos[:, 0] <= x_range[1])
      & (true_ball_pos[:, 1] >= y_range[0])
      & (true_ball_pos[:, 1] <= y_range[1])
    )
    if dropout_probability > 0.0:
      random_visible = (
        torch.rand(env.num_envs, device=env.device) >= dropout_probability
      )
      visible = in_rectangle & random_visible
    else:
      visible = in_rectangle
    visible_float = visible.unsqueeze(1).to(perceived_ball_pos.dtype)

    cache["ball_pos"][new_step] = (perceived_ball_pos * visible_float)[new_step]
    cache["feet"][new_step] = (perceived_feet * visible_float)[new_step]
    cache["visible"][new_step] = visible_float[new_step]
    cache["step"][new_step] = env.episode_length_buf[new_step]

  return cache["ball_pos"], cache["feet"], cache["visible"]


def masked_ball_pos_b(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float] = (0.05, 1.00),
  y_range: tuple[float, float] = (-0.70, 0.70),
  dropout_probability: float = 0.0,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Masked perceived ball XY position in the robot yaw frame."""
  return _shared_masked_ball_visual(
    env,
    x_range,
    y_range,
    dropout_probability,
    bias_range,
    frame_noise_range,
    ball_cfg,
    asset_cfg,
  )[0]


def masked_ball_to_feet_vectors_b(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float] = (0.05, 1.00),
  y_range: tuple[float, float] = (-0.70, 0.70),
  dropout_probability: float = 0.0,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Masked foot-to-ball vectors synchronized with ``masked_ball_pos_b``."""
  return _shared_masked_ball_visual(
    env,
    x_range,
    y_range,
    dropout_probability,
    bias_range,
    frame_noise_range,
    ball_cfg,
    asset_cfg,
  )[1]


def ball_visible_mask(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float] = (0.05, 1.00),
  y_range: tuple[float, float] = (-0.70, 0.70),
  dropout_probability: float = 0.0,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """One when the ball observation is available, otherwise zero."""
  return _shared_masked_ball_visual(
    env,
    x_range,
    y_range,
    dropout_probability,
    bias_range,
    frame_noise_range,
    ball_cfg,
    asset_cfg,
  )[2]


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
