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
  episode_dropout_probability: float,
  bias_range: float,
  frame_noise_range: float,
  ball_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg,
  visibility_rise_alpha: float = 0.20,
  visibility_fall_alpha: float = 0.05,
  sensor_reward_fade_out_s: float | None = None,
  sensor_reward_fade_in_s: float | None = None,
  transition_excluded_standing_command_name: str | None = None,
  transition_dropout_probability: float = 0.0,
  transition_dropout_start_range_s: tuple[float, float] = (2.0, 6.0),
  transition_dropout_duration_range_s: tuple[float, float] = (0.2, 0.8),
  transition_dropout_until_end_probability: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return synchronized masked ball position, foot vectors, and visibility.

  The result is cached for the current control step because the three actor
  observation terms, and their history-group copies, are evaluated separately.
  """
  if not 0.0 <= dropout_probability <= 1.0:
    raise ValueError("dropout_probability must be in [0, 1]")
  if not 0.0 <= episode_dropout_probability <= 1.0:
    raise ValueError("episode_dropout_probability must be in [0, 1]")
  if x_range[0] > x_range[1] or y_range[0] > y_range[1]:
    raise ValueError("ball visibility ranges must be ordered")
  if not 0.0 < visibility_rise_alpha <= 1.0:
    raise ValueError("visibility_rise_alpha must be in (0, 1]")
  if not 0.0 < visibility_fall_alpha <= 1.0:
    raise ValueError("visibility_fall_alpha must be in (0, 1]")
  if sensor_reward_fade_out_s is not None and sensor_reward_fade_out_s <= 0.0:
    raise ValueError("sensor_reward_fade_out_s must be positive")
  if sensor_reward_fade_in_s is not None and sensor_reward_fade_in_s <= 0.0:
    raise ValueError("sensor_reward_fade_in_s must be positive")
  if not 0.0 <= transition_dropout_probability <= 1.0:
    raise ValueError("transition_dropout_probability must be in [0, 1]")
  if not 0.0 <= transition_dropout_until_end_probability <= 1.0:
    raise ValueError("transition_dropout_until_end_probability must be in [0, 1]")
  if (
    transition_dropout_start_range_s[0] < 0.0
    or transition_dropout_start_range_s[0] > transition_dropout_start_range_s[1]
  ):
    raise ValueError(
      "transition_dropout_start_range_s must be non-negative and ordered"
    )
  if (
    transition_dropout_duration_range_s[0] <= 0.0
    or transition_dropout_duration_range_s[0] > transition_dropout_duration_range_s[1]
  ):
    raise ValueError("transition_dropout_duration_range_s must be positive and ordered")

  env_state = vars(env)
  cache = cast(
    dict[str, torch.Tensor] | None,
    env_state.get("_football_masked_ball_visual"),
  )
  valid_cache = (
    isinstance(cache, dict)
    and "episode_hidden" in cache
    and "synthetic_hidden" in cache
    and "visibility_gate" in cache
    and "sensor_gate" in cache
    and cache["step"].shape == env.episode_length_buf.shape
    and cache["ball_pos"].shape == (env.num_envs, 2)
    and cache["episode_hidden"].shape == env.episode_length_buf.shape
  )
  if not valid_cache:
    cache = {
      "step": torch.full_like(env.episode_length_buf, -1),
      "ball_pos": torch.zeros(env.num_envs, 2, device=env.device),
      "feet": torch.zeros(env.num_envs, 4, device=env.device),
      "visible": torch.zeros(env.num_envs, 1, device=env.device),
      "visibility_gate": torch.zeros(env.num_envs, device=env.device),
      "sensor_gate": torch.ones(env.num_envs, device=env.device),
      "sensor_blend_progress": torch.ones(env.num_envs, device=env.device),
      "episode_hidden": torch.zeros(
        env.num_envs,
        dtype=torch.bool,
        device=env.device,
      ),
      "synthetic_hidden": torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
      ),
      "transition_episode": torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
      ),
      "transition_start_step": torch.zeros_like(env.episode_length_buf),
      "transition_end_step": torch.zeros_like(env.episode_length_buf),
    }
    env_state["_football_masked_ball_visual"] = cache
  assert cache is not None

  new_step = cache["step"] != env.episode_length_buf
  if torch.any(new_step):
    reset_mask = new_step & (env.episode_length_buf == 0)
    if torch.any(reset_mask):
      if episode_dropout_probability > 0.0:
        cache["episode_hidden"][reset_mask] = (
          torch.rand(env.num_envs, device=env.device)[reset_mask]
          < episode_dropout_probability
        )
      else:
        cache["episode_hidden"][reset_mask] = False
      reset_ids = torch.where(reset_mask)[0]
      if transition_dropout_probability > 0.0:
        eligible_ids = reset_ids
        if transition_excluded_standing_command_name is not None:
          command_term = env.command_manager.get_term(
            transition_excluded_standing_command_name
          )
          standing = getattr(command_term, "is_standing_env", None)
          if not isinstance(standing, torch.Tensor):
            raise ValueError(
              f"Command {transition_excluded_standing_command_name!r} does not "
              "expose is_standing_env."
            )
          eligible_ids = reset_ids[~standing[reset_ids]]
        cache["transition_episode"][reset_ids] = False
        cache["transition_episode"][eligible_ids] = (
          torch.rand(eligible_ids.numel(), device=env.device)
          < transition_dropout_probability
        )
        start_s = torch.empty(reset_ids.numel(), device=env.device).uniform_(
          *transition_dropout_start_range_s
        )
        duration_s = torch.empty(reset_ids.numel(), device=env.device).uniform_(
          *transition_dropout_duration_range_s
        )
        start_step = torch.ceil(start_s / env.step_dt).to(
          dtype=env.episode_length_buf.dtype
        )
        duration_step = torch.clamp(
          torch.ceil(duration_s / env.step_dt).to(dtype=env.episode_length_buf.dtype),
          min=1,
        )
        until_end = (
          torch.rand(reset_ids.numel(), device=env.device)
          < transition_dropout_until_end_probability
        )
        end_step = start_step + duration_step
        end_step[until_end] = torch.iinfo(env.episode_length_buf.dtype).max
        cache["transition_start_step"][reset_ids] = start_step
        cache["transition_end_step"][reset_ids] = end_step
      else:
        cache["transition_episode"][reset_ids] = False

    synthetic_hidden = (
      cache["transition_episode"]
      & (env.episode_length_buf >= cache["transition_start_step"])
      & (env.episode_length_buf < cache["transition_end_step"])
    )
    cache["synthetic_hidden"][new_step] = synthetic_hidden[new_step]

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
    sensor_hidden = cache["episode_hidden"] | cache["synthetic_hidden"]
    reward_visible = in_rectangle.clone()
    visible = reward_visible & ~sensor_hidden
    if dropout_probability > 0.0:
      random_visible = (
        torch.rand(env.num_envs, device=env.device) >= dropout_probability
      )
      visible &= random_visible
      reward_visible &= random_visible
    visible_float = visible.unsqueeze(1).to(perceived_ball_pos.dtype)

    # Reward-side mode blending uses a short visibility decay.  The Actor still
    # receives the binary mask and its ten-frame history, so this does not alter
    # the exported observation contract.
    visibility_gate = cache["visibility_gate"]
    target_visibility = reward_visible[new_step].to(visibility_gate.dtype)
    alpha = torch.where(
      target_visibility > visibility_gate[new_step],
      visibility_rise_alpha,
      visibility_fall_alpha,
    )
    visibility_gate[new_step] += alpha * (target_visibility - visibility_gate[new_step])
    visibility_gate[reset_mask] = reward_visible[reset_mask].to(visibility_gate.dtype)

    sensor_gate = cache["sensor_gate"]
    target_sensor = (~sensor_hidden[new_step]).to(sensor_gate.dtype)
    if sensor_reward_fade_out_s is None:
      sensor_alpha = torch.where(
        target_sensor > sensor_gate[new_step],
        visibility_rise_alpha,
        visibility_fall_alpha,
      )
      sensor_gate[new_step] += sensor_alpha * (target_sensor - sensor_gate[new_step])
      sensor_gate[reset_mask] = target_sensor[reset_mask[new_step]]
    else:
      fade_in_s = sensor_reward_fade_in_s or sensor_reward_fade_out_s
      progress = cache["sensor_blend_progress"]
      progress_step = torch.where(
        sensor_hidden[new_step],
        torch.full_like(target_sensor, -env.step_dt / sensor_reward_fade_out_s),
        torch.full_like(target_sensor, env.step_dt / fade_in_s),
      )
      progress[new_step] = torch.clamp(
        progress[new_step] + progress_step,
        min=0.0,
        max=1.0,
      )
      progress[reset_mask] = target_sensor[reset_mask[new_step]]
      sensor_gate[new_step] = progress[new_step].square() * (
        3.0 - 2.0 * progress[new_step]
      )

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
  episode_dropout_probability: float = 0.0,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  visibility_rise_alpha: float = 0.20,
  visibility_fall_alpha: float = 0.05,
  sensor_reward_fade_out_s: float | None = None,
  sensor_reward_fade_in_s: float | None = None,
  transition_excluded_standing_command_name: str | None = None,
  transition_dropout_probability: float = 0.0,
  transition_dropout_start_range_s: tuple[float, float] = (2.0, 6.0),
  transition_dropout_duration_range_s: tuple[float, float] = (0.2, 0.8),
  transition_dropout_until_end_probability: float = 0.0,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Masked perceived ball XY position in the robot yaw frame."""
  return _shared_masked_ball_visual(
    env,
    x_range,
    y_range,
    dropout_probability,
    episode_dropout_probability,
    bias_range,
    frame_noise_range,
    ball_cfg,
    asset_cfg,
    visibility_rise_alpha,
    visibility_fall_alpha,
    sensor_reward_fade_out_s,
    sensor_reward_fade_in_s,
    transition_excluded_standing_command_name,
    transition_dropout_probability,
    transition_dropout_start_range_s,
    transition_dropout_duration_range_s,
    transition_dropout_until_end_probability,
  )[0]


def masked_ball_to_feet_vectors_b(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float] = (0.05, 1.00),
  y_range: tuple[float, float] = (-0.70, 0.70),
  dropout_probability: float = 0.0,
  episode_dropout_probability: float = 0.0,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  visibility_rise_alpha: float = 0.20,
  visibility_fall_alpha: float = 0.05,
  sensor_reward_fade_out_s: float | None = None,
  sensor_reward_fade_in_s: float | None = None,
  transition_excluded_standing_command_name: str | None = None,
  transition_dropout_probability: float = 0.0,
  transition_dropout_start_range_s: tuple[float, float] = (2.0, 6.0),
  transition_dropout_duration_range_s: tuple[float, float] = (0.2, 0.8),
  transition_dropout_until_end_probability: float = 0.0,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Masked foot-to-ball vectors synchronized with ``masked_ball_pos_b``."""
  return _shared_masked_ball_visual(
    env,
    x_range,
    y_range,
    dropout_probability,
    episode_dropout_probability,
    bias_range,
    frame_noise_range,
    ball_cfg,
    asset_cfg,
    visibility_rise_alpha,
    visibility_fall_alpha,
    sensor_reward_fade_out_s,
    sensor_reward_fade_in_s,
    transition_excluded_standing_command_name,
    transition_dropout_probability,
    transition_dropout_start_range_s,
    transition_dropout_duration_range_s,
    transition_dropout_until_end_probability,
  )[1]


def ball_visible_mask(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float] = (0.05, 1.00),
  y_range: tuple[float, float] = (-0.70, 0.70),
  dropout_probability: float = 0.0,
  episode_dropout_probability: float = 0.0,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  visibility_rise_alpha: float = 0.20,
  visibility_fall_alpha: float = 0.05,
  sensor_reward_fade_out_s: float | None = None,
  sensor_reward_fade_in_s: float | None = None,
  transition_excluded_standing_command_name: str | None = None,
  transition_dropout_probability: float = 0.0,
  transition_dropout_start_range_s: tuple[float, float] = (2.0, 6.0),
  transition_dropout_duration_range_s: tuple[float, float] = (0.2, 0.8),
  transition_dropout_until_end_probability: float = 0.0,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """One when the ball observation is available, otherwise zero."""
  return _shared_masked_ball_visual(
    env,
    x_range,
    y_range,
    dropout_probability,
    episode_dropout_probability,
    bias_range,
    frame_noise_range,
    ball_cfg,
    asset_cfg,
    visibility_rise_alpha,
    visibility_fall_alpha,
    sensor_reward_fade_out_s,
    sensor_reward_fade_in_s,
    transition_excluded_standing_command_name,
    transition_dropout_probability,
    transition_dropout_start_range_s,
    transition_dropout_duration_range_s,
    transition_dropout_until_end_probability,
  )[2]


def masked_ball_features_b(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float] = (0.05, 1.00),
  y_range: tuple[float, float] = (-0.70, 0.70),
  dropout_probability: float = 0.0,
  episode_dropout_probability: float = 0.0,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  visibility_rise_alpha: float = 0.20,
  visibility_fall_alpha: float = 0.05,
  sensor_reward_fade_out_s: float | None = None,
  sensor_reward_fade_in_s: float | None = None,
  transition_excluded_standing_command_name: str | None = None,
  transition_dropout_probability: float = 0.0,
  transition_dropout_start_range_s: tuple[float, float] = (2.0, 6.0),
  transition_dropout_duration_range_s: tuple[float, float] = (0.2, 0.8),
  transition_dropout_until_end_probability: float = 0.0,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Return the complete 7-D ball stream for one shared delay buffer.

  Packing position, both foot vectors, and visibility into one observation term
  guarantees that all seven values are selected from the same delayed frame.
  """
  ball_pos, feet, visible = _shared_masked_ball_visual(
    env,
    x_range,
    y_range,
    dropout_probability,
    episode_dropout_probability,
    bias_range,
    frame_noise_range,
    ball_cfg,
    asset_cfg,
    visibility_rise_alpha,
    visibility_fall_alpha,
    sensor_reward_fade_out_s,
    sensor_reward_fade_in_s,
    transition_excluded_standing_command_name,
    transition_dropout_probability,
    transition_dropout_start_range_s,
    transition_dropout_duration_range_s,
    transition_dropout_until_end_probability,
  )
  return torch.cat((ball_pos, feet, visible), dim=-1)


def episode_ball_observation_hidden(
  env: ManagerBasedRlEnv,
  x_range: tuple[float, float] = (0.05, 1.00),
  y_range: tuple[float, float] = (-0.70, 0.70),
  dropout_probability: float = 0.0,
  episode_dropout_probability: float = 0.0,
  bias_range: float = 0.10,
  frame_noise_range: float = 0.06,
  visibility_rise_alpha: float = 0.20,
  visibility_fall_alpha: float = 0.05,
  sensor_reward_fade_out_s: float | None = None,
  sensor_reward_fade_in_s: float | None = None,
  transition_excluded_standing_command_name: str | None = None,
  transition_dropout_probability: float = 0.0,
  transition_dropout_start_range_s: tuple[float, float] = (2.0, 6.0),
  transition_dropout_duration_range_s: tuple[float, float] = (0.2, 0.8),
  transition_dropout_until_end_probability: float = 0.0,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Privileged Critic flag for whole-episode synthetic ball blindness.

  Calling the shared visual path makes this term independent of observation
  group evaluation order while preserving the exact Actor dropout sample.
  """
  _shared_masked_ball_visual(
    env,
    x_range,
    y_range,
    dropout_probability,
    episode_dropout_probability,
    bias_range,
    frame_noise_range,
    ball_cfg,
    asset_cfg,
    visibility_rise_alpha,
    visibility_fall_alpha,
    sensor_reward_fade_out_s,
    sensor_reward_fade_in_s,
    transition_excluded_standing_command_name,
    transition_dropout_probability,
    transition_dropout_start_range_s,
    transition_dropout_duration_range_s,
    transition_dropout_until_end_probability,
  )
  cache = vars(env)["_football_masked_ball_visual"]
  return cache["episode_hidden"].to(dtype=torch.float32).unsqueeze(1)


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
