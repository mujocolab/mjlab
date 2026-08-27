from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  wrap_to_pi,
  yaw_quat,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _soft_deadband(
  value: torch.Tensor,
  deadband: torch.Tensor,
) -> torch.Tensor:
  magnitude = torch.relu(torch.abs(value) - deadband)
  return torch.sign(value) * magnitude


def _limit_vector_change(
  current: torch.Tensor,
  target: torch.Tensor,
  max_change: float,
) -> torch.Tensor:
  difference = target - current
  difference_norm = torch.linalg.vector_norm(
    difference,
    dim=1,
    keepdim=True,
  )
  scale = torch.clamp(
    max_change / difference_norm.clamp_min(1e-6),
    max=1.0,
  )
  return current + scale * difference


def _limit_vector_change_asymmetric(
  current: torch.Tensor,
  target: torch.Tensor,
  max_increase: float,
  max_decrease: float,
) -> torch.Tensor:
  difference = target - current
  is_speeding_up = (current * target >= 0.0) & (torch.abs(target) > torch.abs(current))
  max_change = torch.where(
    is_speeding_up,
    torch.full_like(difference, max_increase),
    torch.full_like(difference, max_decrease),
  )
  return current + torch.clamp(difference, -max_change, max_change)


class BallRelativeVelocityReference:
  """Generate a smooth robot velocity reference from observed football motion."""

  def __init__(
    self,
    cfg: BallRelativeVelocityReferenceCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    self.cfg = cfg
    self.env = env
    self.robot: Entity = env.scene[cfg.robot_entity_name]
    self.ball: Entity = env.scene[cfg.ball_entity_name]
    self.device = env.device

    self.anchor = torch.tensor(cfg.anchor, device=self.device)
    self.position_deadband = torch.tensor(
      cfg.position_deadband,
      device=self.device,
    )
    self.velocity_deadband = torch.tensor(
      cfg.velocity_deadband,
      device=self.device,
    )
    self.position_gain = torch.tensor(cfg.position_gain, device=self.device)
    self.velocity_gain = torch.tensor(cfg.velocity_gain, device=self.device)

    shape = (env.num_envs, 2)
    self.fixed_position_bias = torch.zeros(shape, device=self.device)
    self.filtered_position = torch.zeros(shape, device=self.device)
    self.previous_filtered_position = torch.zeros(shape, device=self.device)
    self.filtered_relative_velocity = torch.zeros(shape, device=self.device)
    self.base_velocity = torch.zeros(shape, device=self.device)
    self.reference_velocity = torch.zeros(shape, device=self.device)
    self.initialized = torch.zeros(
      env.num_envs,
      dtype=torch.bool,
      device=self.device,
    )

  def reset(self, env_ids: torch.Tensor, user_command_b: torch.Tensor) -> None:
    """Reset the explicitly selected environments."""
    if len(env_ids) == 0:
      return
    self.fixed_position_bias[env_ids].uniform_(
      -self.cfg.fixed_position_bias_range,
      self.cfg.fixed_position_bias_range,
    )
    self.base_velocity[env_ids] = user_command_b[env_ids, :2]
    self.reference_velocity[env_ids] = user_command_b[env_ids, :2]
    self.initialized[env_ids] = False

  def _observe_ball_position(self) -> torch.Tensor:
    ball_relative_w = self.ball.data.root_link_pos_w - self.robot.data.root_link_pos_w
    position = quat_apply_inverse(
      yaw_quat(self.robot.data.root_link_quat_w),
      ball_relative_w,
    )[:, :2]
    if self.cfg.frame_position_noise_range > 0.0:
      position = position + torch.empty_like(position).uniform_(
        -self.cfg.frame_position_noise_range,
        self.cfg.frame_position_noise_range,
      )
    return position + self.fixed_position_bias

  def update(
    self,
    user_command_b: torch.Tensor,
    dt: float,
    velocity_x_range: tuple[float, float],
    velocity_y_range: tuple[float, float],
  ) -> torch.Tensor:
    observed_position = self._observe_ball_position()
    new_envs = ~self.initialized
    if torch.any(new_envs):
      self.filtered_position[new_envs] = observed_position[new_envs]
      self.previous_filtered_position[new_envs] = observed_position[new_envs]
      self.filtered_relative_velocity[new_envs] = 0.0
      self.base_velocity[new_envs] = user_command_b[new_envs, :2]
      self.reference_velocity[new_envs] = user_command_b[new_envs, :2]
      self.initialized[new_envs] = True

    self.filtered_position.lerp_(
      observed_position,
      self.cfg.position_filter_alpha,
    )
    raw_relative_velocity = (
      self.filtered_position - self.previous_filtered_position
    ) / dt
    self.previous_filtered_position.copy_(self.filtered_position)

    raw_speed = torch.linalg.vector_norm(
      raw_relative_velocity,
      dim=1,
      keepdim=True,
    )
    raw_relative_velocity *= torch.clamp(
      self.cfg.max_relative_speed / raw_speed.clamp_min(1e-6),
      max=1.0,
    )
    self.filtered_relative_velocity.lerp_(
      raw_relative_velocity,
      self.cfg.velocity_filter_alpha,
    )

    position_error = _soft_deadband(
      self.filtered_position - self.anchor,
      self.position_deadband,
    )
    relative_velocity = _soft_deadband(
      self.filtered_relative_velocity,
      self.velocity_deadband,
    )
    correction = (
      self.position_gain * position_error + self.velocity_gain * relative_velocity
    )
    correction[:, 0].clamp_(-self.cfg.max_correction[0], self.cfg.max_correction[0])
    correction[:, 1].clamp_(-self.cfg.max_correction[1], self.cfg.max_correction[1])

    self.base_velocity = _limit_vector_change(
      self.base_velocity,
      user_command_b[:, :2],
      self.cfg.base_acceleration * dt,
    )
    target = self.base_velocity + correction
    target[:, 0].clamp_(*velocity_x_range)
    target[:, 1].clamp_(*velocity_y_range)
    self.reference_velocity = _limit_vector_change_asymmetric(
      self.reference_velocity,
      target,
      self.cfg.max_acceleration_up * dt,
      self.cfg.max_acceleration_down * dt,
    )

    self.env.extras["log"]["Metrics/command_correction_xy"] = torch.mean(
      torch.linalg.vector_norm(correction, dim=1)
    )
    self.env.extras["log"]["Metrics/command_reference_error_xy"] = torch.mean(
      torch.linalg.vector_norm(
        self.reference_velocity - user_command_b[:, :2],
        dim=1,
      )
    )
    return self.reference_velocity


def _minimum_jerk(progress: torch.Tensor) -> torch.Tensor:
  """Return minimum-jerk interpolation weights on the unit interval."""
  progress = progress.clamp(0.0, 1.0)
  return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


class StopSkillVelocityReference:
  """Generate vectorized rise-and-fall stop references for training."""

  IDLE = 0
  RISE = 1
  FALL = 2

  def __init__(
    self,
    cfg: StopSkillVelocityReferenceCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    self.cfg = cfg
    self.env = env
    self.device = env.device

    shape = (env.num_envs, 2)
    self.command_history = torch.zeros(
      env.num_envs,
      cfg.trigger_window + 1,
      device=self.device,
    )
    self.state = torch.full(
      (env.num_envs,),
      self.IDLE,
      dtype=torch.long,
      device=self.device,
    )
    self.armed = torch.ones(
      env.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.condition_count = torch.zeros(
      env.num_envs,
      dtype=torch.long,
      device=self.device,
    )
    self.elapsed = torch.zeros(env.num_envs, device=self.device)
    self.start_reference = torch.zeros(shape, device=self.device)
    self.final_reference = torch.zeros(shape, device=self.device)
    self.peak_reference = torch.zeros(shape, device=self.device)
    self.green_at_peak = torch.zeros(shape, device=self.device)
    self.target_reference = torch.zeros(shape, device=self.device)
    self.reference_velocity = torch.zeros(shape, device=self.device)
    self.initialized = torch.zeros(
      env.num_envs,
      dtype=torch.bool,
      device=self.device,
    )

  def _normalized_command_speed(self, command: torch.Tensor) -> torch.Tensor:
    return (
      torch.linalg.vector_norm(command[:, :2], dim=1) / self.cfg.maximum_velocity
    ).clamp(0.0, 1.0)

  def reset(self, env_ids: torch.Tensor, user_command_b: torch.Tensor) -> None:
    """Reset the explicitly selected environments."""
    if len(env_ids) == 0:
      return
    normalized_speed = self._normalized_command_speed(user_command_b[env_ids])
    self.command_history[env_ids] = normalized_speed.unsqueeze(1)
    self.state[env_ids] = self.IDLE
    self.armed[env_ids] = True
    self.condition_count[env_ids] = 0
    self.elapsed[env_ids] = 0.0
    initial_reference = user_command_b[env_ids, :2]
    self.start_reference[env_ids] = initial_reference
    self.final_reference[env_ids] = initial_reference
    self.peak_reference[env_ids] = initial_reference
    self.green_at_peak[env_ids] = initial_reference
    self.target_reference[env_ids] = initial_reference
    self.reference_velocity[env_ids] = initial_reference
    self.initialized[env_ids] = True

  def _green_reference(
    self,
    env_ids: torch.Tensor,
    elapsed: torch.Tensor,
  ) -> torch.Tensor:
    total_duration = self.cfg.rise_duration + self.cfg.fall_duration
    weight = _minimum_jerk(elapsed / total_duration).unsqueeze(1)
    return self.start_reference[env_ids] + weight * (
      self.final_reference[env_ids] - self.start_reference[env_ids]
    )

  def _start(
    self,
    env_ids: torch.Tensor,
    user_command_b: torch.Tensor,
    velocity_x_range: tuple[float, float],
    velocity_y_range: tuple[float, float],
  ) -> None:
    self.state[env_ids] = self.RISE
    self.elapsed[env_ids] = 0.0
    self.start_reference[env_ids] = self.reference_velocity[env_ids]
    self.final_reference[env_ids] = user_command_b[env_ids, :2]
    planar_speed = torch.linalg.vector_norm(
      self.start_reference[env_ids],
      dim=1,
      keepdim=True,
    )
    direction = self.start_reference[env_ids] / planar_speed.clamp_min(1e-6)
    peak_speed = (planar_speed + self.cfg.rise_amplitude).clamp(
      max=self.cfg.maximum_velocity
    )
    peak_reference = direction * peak_speed
    peak_reference[:, 0].clamp_(*velocity_x_range)
    peak_reference[:, 1].clamp_(*velocity_y_range)
    self.peak_reference[env_ids] = peak_reference
    self.green_at_peak[env_ids] = self._green_reference(
      env_ids,
      torch.full_like(self.elapsed[env_ids], self.cfg.rise_duration),
    )

  def update(
    self,
    user_command_b: torch.Tensor,
    dt: float,
    velocity_x_range: tuple[float, float],
    velocity_y_range: tuple[float, float],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the policy reference and monotonic football target."""
    if dt <= 0.0:
      raise ValueError("Stop-skill update period must be positive.")

    normalized_command = self._normalized_command_speed(user_command_b)
    old_command = self.command_history[:, 0]
    window_drop = old_command - normalized_command
    window_acceleration = (normalized_command - old_command) / (
      self.cfg.trigger_window * dt
    )
    condition = (window_acceleration < -self.cfg.acceleration_threshold) & (
      window_drop > self.cfg.minimum_command_drop
    )
    self.condition_count = torch.where(
      condition,
      self.condition_count + 1,
      torch.zeros_like(self.condition_count),
    )
    idle = self.state == self.IDLE
    triggered = (
      self.armed & idle & (self.condition_count >= self.cfg.persistence_frames)
    )
    trigger_ids = triggered.nonzero(as_tuple=False).flatten()
    if len(trigger_ids) > 0:
      self.armed[trigger_ids] = False
      self.condition_count[trigger_ids] = 0
      self._start(
        trigger_ids,
        user_command_b,
        velocity_x_range,
        velocity_y_range,
      )

    stable_idle = idle & (
      torch.abs(window_acceleration) < self.cfg.rearm_acceleration_threshold
    )
    self.armed[stable_idle] = True
    self.command_history = torch.roll(self.command_history, shifts=-1, dims=1)
    self.command_history[:, -1] = normalized_command

    idle = self.state == self.IDLE
    pending = idle & (self.condition_count > 0)
    following = idle & ~pending
    self.final_reference[following] = user_command_b[following, :2]
    self.target_reference[following] = user_command_b[following, :2]
    self.reference_velocity[following] = user_command_b[following, :2]

    active = self.state != self.IDLE
    active_ids = active.nonzero(as_tuple=False).flatten()
    if len(active_ids) > 0:
      self.elapsed[active_ids] += dt
      green_reference = self._green_reference(
        active_ids,
        self.elapsed[active_ids],
      )
      self.target_reference[active_ids] = green_reference

      rising = active & (self.elapsed <= self.cfg.rise_duration)
      rising_ids = rising.nonzero(as_tuple=False).flatten()
      if len(rising_ids) > 0:
        weight = _minimum_jerk(
          self.elapsed[rising_ids] / self.cfg.rise_duration
        ).unsqueeze(1)
        self.reference_velocity[rising_ids] = self.start_reference[
          rising_ids
        ] + weight * (
          self.peak_reference[rising_ids] - self.start_reference[rising_ids]
        )

      falling = active & ~rising
      falling_ids = falling.nonzero(as_tuple=False).flatten()
      if len(falling_ids) > 0:
        self.state[falling_ids] = self.FALL
        fall_elapsed = self.elapsed[falling_ids] - self.cfg.rise_duration
        weight = _minimum_jerk(fall_elapsed / self.cfg.fall_duration).unsqueeze(1)
        self.reference_velocity[falling_ids] = self.target_reference[falling_ids] + (
          self.peak_reference[falling_ids] - self.green_at_peak[falling_ids]
        ) * (1.0 - weight)
        finished = fall_elapsed >= self.cfg.fall_duration
        finished_ids = falling_ids[finished]
        if len(finished_ids) > 0:
          self.state[finished_ids] = self.IDLE
          self.reference_velocity[finished_ids] = self.target_reference[finished_ids]

    self.env.extras["log"]["Metrics/stop_skill_active"] = active.float().mean()
    self.env.extras["log"]["Metrics/stop_skill_reference_error_xy"] = torch.mean(
      torch.linalg.vector_norm(
        self.reference_velocity - self.target_reference,
        dim=1,
      )
    )
    return self.reference_velocity, self.target_reference


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.user_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)
    self.zero_ramp_active = torch.zeros_like(self.is_heading_env)
    self.zero_ramp_start_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.zero_ramp_duration = torch.ones(self.num_envs, device=self.device)
    self.zero_ramp_elapsed = torch.zeros(self.num_envs, device=self.device)
    self.ball_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.ball_reference = (
      BallRelativeVelocityReference(cfg.ball_relative_velocity_reference, env)
      if cfg.ball_relative_velocity_reference is not None
      else None
    )
    self.stop_skill_reference = (
      StopSkillVelocityReference(cfg.stop_skill_velocity_reference, env)
      if cfg.stop_skill_velocity_reference is not None
      else None
    )

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  @property
  def user_command(self) -> torch.Tensor:
    """Return the unmodified task command sampled for the user."""
    return self.user_command_b

  @property
  def ball_command(self) -> torch.Tensor:
    """Return the monotonic velocity target used by football rewards."""
    return self.ball_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.user_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
        dim=-1,
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.user_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    previous_command_b = self.vel_command_b[env_ids].clone()
    # CommandTerm.reset() clears command_counter before calling this method,
    # whereas periodic command resampling leaves it positive. The episode length
    # buffer cannot distinguish these cases because the environment clears it
    # only after CommandManager.reset() returns.
    episode_reset_ids = env_ids[self.command_counter[env_ids] == 0]
    r = torch.empty(len(env_ids), device=self.device)
    self.user_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.user_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.user_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    standing_sample_ids = (
      episode_reset_ids if self.cfg.standing_mode_per_episode else env_ids
    )
    self.is_standing_env[standing_sample_ids] = (
      torch.empty(len(standing_sample_ids), device=self.device).uniform_(0.0, 1.0)
      <= self.cfg.rel_standing_envs
    )

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.user_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      self.user_command_b[fwd_ids, 0] = (
        self.user_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      self.user_command_b[fwd_ids, 1] = 0.0
      self.user_command_b[fwd_ids, 2] = 0.0

    standing_ids = env_ids[self.is_standing_env[env_ids]]
    self.user_command_b[standing_ids] = 0.0

    self.zero_ramp_active[env_ids] = False
    if (
      self.ball_reference is None and self.cfg.zero_command_ramp_time_range is not None
    ):
      is_mid_episode = self.command_counter[env_ids] > 0
      has_previous_motion = torch.linalg.vector_norm(previous_command_b, dim=1) > 1e-6
      ramp_mask = self.is_standing_env[env_ids] & is_mid_episode & has_previous_motion
      ramp_ids = env_ids[ramp_mask]
      if len(ramp_ids) > 0:
        self.zero_ramp_active[ramp_ids] = True
        self.zero_ramp_start_b[ramp_ids] = previous_command_b[ramp_mask]
        self.zero_ramp_elapsed[ramp_ids] = 0.0
        self.zero_ramp_duration[ramp_ids] = self.zero_ramp_duration[ramp_ids].uniform_(
          *self.cfg.zero_command_ramp_time_range
        )

    if self.ball_reference is not None:
      self.ball_reference.reset(episode_reset_ids, self.user_command_b)
    if self.stop_skill_reference is not None:
      self.stop_skill_reference.reset(episode_reset_ids, self.user_command_b)
    # CommandManager.reset() returns observations before the next compute() call.
    # Synchronize the command exposed to the policy immediately so the first
    # observation of a new episode cannot contain the previous episode's command.
    if len(episode_reset_ids) > 0:
      self.vel_command_b[episode_reset_ids] = self.user_command_b[episode_reset_ids]
      self.ball_command_b[episode_reset_ids] = self.user_command_b[episode_reset_ids]

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.user_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.user_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.user_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.user_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.user_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.user_command_b[standing_env_ids] = 0.0
    self.vel_command_w[standing_env_ids] = 0.0

    if self.ball_reference is not None:
      self.vel_command_b[:, :2] = self.ball_reference.update(
        self.user_command_b,
        self._env.step_dt,
        self.cfg.ranges.lin_vel_x,
        self.cfg.ranges.lin_vel_y,
      )
      self.vel_command_b[:, 2] = self.user_command_b[:, 2]
      self.ball_command_b.copy_(self.user_command_b)
      return

    if self.stop_skill_reference is not None:
      policy_reference, ball_target = self.stop_skill_reference.update(
        self.user_command_b,
        self._env.step_dt,
        self.cfg.ranges.lin_vel_x,
        self.cfg.ranges.lin_vel_y,
      )
      self.vel_command_b[:, :2] = policy_reference
      self.vel_command_b[:, 2] = self.user_command_b[:, 2]
      self.ball_command_b[:, :2] = ball_target
      self.ball_command_b[:, 2] = self.user_command_b[:, 2]
      return

    self.vel_command_b.copy_(self.user_command_b)
    self.ball_command_b.copy_(self.user_command_b)
    if self.cfg.zero_command_ramp_time_range is None:
      return

    immediate_stop_ids = standing_env_ids[~self.zero_ramp_active[standing_env_ids]]
    self.vel_command_b[immediate_stop_ids, :] = 0.0
    ramp_ids = self.zero_ramp_active.nonzero(as_tuple=False).flatten()
    if len(ramp_ids) > 0:
      self.zero_ramp_elapsed[ramp_ids] += self._env.step_dt
      progress = torch.clamp(
        self.zero_ramp_elapsed[ramp_ids] / self.zero_ramp_duration[ramp_ids],
        max=1.0,
      )
      self.vel_command_b[ramp_ids] = self.zero_ramp_start_b[ramp_ids] * (
        1.0 - progress.unsqueeze(1)
      )
      finished_ids = ramp_ids[progress >= 1.0]
      self.vel_command_b[finished_ids, :] = 0.0
      self.zero_ramp_active[finished_ids] = False

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class BallRelativeVelocityReferenceCfg:
  """Configuration for an explicit football-aware velocity reference."""

  robot_entity_name: str = "robot"
  ball_entity_name: str = "ball"
  anchor: tuple[float, float] = (0.25, 0.0)
  position_deadband: tuple[float, float] = (0.07, 0.06)
  velocity_deadband: tuple[float, float] = (0.05, 0.05)
  position_gain: tuple[float, float] = (0.3, 0.5)
  velocity_gain: tuple[float, float] = (0.6, 0.6)
  max_correction: tuple[float, float] = (0.4, 0.25)
  base_acceleration: float = 0.4
  max_acceleration_up: float = 0.8
  max_acceleration_down: float = 0.5
  max_relative_speed: float = 2.0
  position_filter_alpha: float = 0.2
  velocity_filter_alpha: float = 0.1
  fixed_position_bias_range: float = 0.10
  frame_position_noise_range: float = 0.06

  def __post_init__(self) -> None:
    if not 0.0 < self.position_filter_alpha <= 1.0:
      raise ValueError("position_filter_alpha must be in (0, 1]")
    if not 0.0 < self.velocity_filter_alpha <= 1.0:
      raise ValueError("velocity_filter_alpha must be in (0, 1]")
    acceleration_limits = (
      self.base_acceleration,
      self.max_acceleration_up,
      self.max_acceleration_down,
    )
    if any(value <= 0.0 for value in acceleration_limits):
      raise ValueError("acceleration limits must be positive")
    if self.max_relative_speed <= 0.0:
      raise ValueError("velocity and acceleration limits must be positive")
    if any(value < 0.0 for value in self.position_deadband):
      raise ValueError("position_deadband must be non-negative")
    if any(value < 0.0 for value in self.velocity_deadband):
      raise ValueError("velocity_deadband must be non-negative")
    if any(value <= 0.0 for value in self.max_correction):
      raise ValueError("max_correction must be positive")
    if self.fixed_position_bias_range < 0.0:
      raise ValueError("fixed_position_bias_range must be non-negative")
    if self.frame_position_noise_range < 0.0:
      raise ValueError("frame_position_noise_range must be non-negative")


@dataclass(kw_only=True)
class StopSkillVelocityReferenceCfg:
  """Configuration for a minimum-jerk training stop skill."""

  maximum_velocity: float = 1.0
  rise_amplitude: float = 0.2
  rise_duration: float = 0.3
  fall_duration: float = 0.3
  trigger_window: int = 5
  acceleration_threshold: float = 1.0
  minimum_command_drop: float = 0.12
  persistence_frames: int = 2
  rearm_acceleration_threshold: float = 0.2

  def __post_init__(self) -> None:
    positive_values = (
      self.maximum_velocity,
      self.rise_duration,
      self.fall_duration,
      self.acceleration_threshold,
    )
    if any(value <= 0.0 for value in positive_values):
      raise ValueError(
        "Stop-skill velocity, durations, and threshold must be positive."
      )
    if self.rise_amplitude < 0.0:
      raise ValueError("Stop-skill rise amplitude must be non-negative.")
    if self.trigger_window < 1 or self.persistence_frames < 1:
      raise ValueError("Stop-skill window and persistence must be positive.")
    normalized_thresholds = (
      self.minimum_command_drop,
      self.rearm_acceleration_threshold,
    )
    if any(value < 0.0 for value in normalized_thresholds):
      raise ValueError("Stop-skill normalized thresholds must be non-negative.")


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  standing_mode_per_episode: bool = False
  """Keep the sampled standing mode fixed across periodic command resampling."""
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  rel_forward_envs: float = 0.0
  """Fraction of environments that receive forward-only commands (positive
  lin_vel_x, zero lin_vel_y and ang_vel_z). Increases training coverage for
  straight-line walking, which is important for stair climbing."""
  init_velocity_prob: float = 0.0
  zero_command_ramp_time_range: tuple[float, float] | None = None
  """Optional linear ramp duration when a moving command resamples to standing.

  Initial standing commands remain zero immediately. Only mid-episode
  moving-to-standing transitions are ramped.
  """
  ball_relative_velocity_reference: BallRelativeVelocityReferenceCfg | None = None
  """Optional explicit ball-relative correction of the policy velocity command."""
  stop_skill_velocity_reference: StopSkillVelocityReferenceCfg | None = None
  """Optional rise-and-fall stop reference with a monotonic football target."""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
    if self.zero_command_ramp_time_range is not None:
      ramp_min, ramp_max = self.zero_command_ramp_time_range
      if ramp_min <= 0.0 or ramp_min > ramp_max:
        raise ValueError(
          "zero_command_ramp_time_range must be positive and ordered, "
          f"got {self.zero_command_ramp_time_range}."
        )
    if (
      self.zero_command_ramp_time_range is not None
      and self.ball_relative_velocity_reference is not None
    ):
      raise ValueError(
        "zero_command_ramp_time_range and ball_relative_velocity_reference "
        "cannot be enabled together"
      )
    reference_count = sum(
      reference is not None
      for reference in (
        self.zero_command_ramp_time_range,
        self.ball_relative_velocity_reference,
        self.stop_skill_velocity_reference,
      )
    )
    if reference_count > 1:
      raise ValueError(
        "Only one zero ramp, ball-relative reference, or stop-skill reference "
        "can be enabled"
      )
