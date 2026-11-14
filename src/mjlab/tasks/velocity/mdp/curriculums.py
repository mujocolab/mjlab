from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  promotion_fraction: float = 0.8,
  min_commanded_distance: float = 1.0,
) -> dict[str, torch.Tensor]:
  """Promote or demote envs by distance walked relative to distance commanded.

  Envs commanded less than min_commanded_distance meters are never promoted, so
  drift cannot promote an env that was asked to stand still.
  """
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, UniformVelocityCommand)

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - command_term.episode_start_pos_w[env_ids],
    dim=1,
  )

  # Episodes that ended early owe the time left at the current command, so that
  # falling does not shrink the requirement.
  time_left = env.max_episode_length_s - env.episode_length_buf[env_ids] * env.step_dt
  commanded_distance = torch.norm(
    command_term.commanded_displacement_w[env_ids], dim=1
  ) + torch.norm(command_term.command[env_ids, :2], dim=1) * time_left.clamp(min=0.0)

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > torch.clamp(
    commanded_distance * promotion_fraction, max=terrain_generator.size[0] / 2
  )
  move_up &= commanded_distance >= min_commanded_distance

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = distance < commanded_distance * 0.5
  move_down &= ~move_up

  # On the initial reset (before any env step) the robot is still at its spawn
  # pose rather than a walked-to position, so ``distance`` is meaningless and
  # would spuriously promote every env from level 0 to 1, ignoring
  # ``max_init_terrain_level``. Freeze levels on that first reset.
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class commands_vel_adaptive:
  """Widen ``lin_vel_x`` command range once episodic tracking reward meets a target.

  Tracks an exponential moving average (EMA) of finished-episode reward as a
  fraction of the reward term's maximum full-episode return, and widens
  ``lin_vel_x`` by ``delta`` toward ``max_lin_vel_x`` at most once every
  ``update_interval_steps`` steps (default: ``env.max_episode_length``). The
  command range never narrows.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    command_name: str = params["command_name"]
    max_lin_vel_x: tuple[float, float] = params["max_lin_vel_x"]
    reward_name: str = params.get("reward_name", "track_linear_velocity")
    reward_fraction: float = params.get("reward_fraction", 0.8)
    delta: float = params.get("delta", 0.1)
    ema_alpha: float = params.get("ema_alpha", 0.1)
    update_interval_steps: int | None = params.get("update_interval_steps", None)

    command_term = env.command_manager.get_term(command_name)
    if command_term is None or not isinstance(
      command_term.cfg, UniformVelocityCommandCfg
    ):
      raise TypeError(
        f"Command term '{command_name}' must use UniformVelocityCommandCfg."
      )
    self._command_cfg: UniformVelocityCommandCfg = command_term.cfg

    self._reward_name = reward_name
    self._reward_cfg = env.reward_manager.get_term_cfg(reward_name)
    if self._reward_cfg.weight <= 0.0:
      raise ValueError(
        f"Reward term '{reward_name}' must have positive weight, "
        f"got {self._reward_cfg.weight}."
      )

    min_x, max_x = max_lin_vel_x
    cur_lo, cur_hi = self._command_cfg.ranges.lin_vel_x
    if min_x > max_x:
      raise ValueError(
        f"max_lin_vel_x must satisfy min <= max, got ({min_x}, {max_x})."
      )
    if min_x > cur_lo or max_x < cur_hi:
      raise ValueError(
        f"max_lin_vel_x ({min_x}, {max_x}) must enclose initial "
        f"lin_vel_x ({cur_lo}, {cur_hi})."
      )
    if delta <= 0.0:
      raise ValueError(f"delta must be positive, got {delta}.")
    if not (0.0 < reward_fraction <= 1.0):
      raise ValueError(f"reward_fraction must be in (0, 1], got {reward_fraction}.")
    if not (0.0 < ema_alpha <= 1.0):
      raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}.")

    interval = (
      env.max_episode_length if update_interval_steps is None else update_interval_steps
    )
    if interval < 1:
      raise ValueError(f"update_interval_steps must be >= 1, got {interval}.")

    self._interval = interval
    self._ema: float | None = None
    self._last_update_step = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    command_name: str,
    max_lin_vel_x: tuple[float, float],
    reward_name: str = "track_linear_velocity",
    reward_fraction: float = 0.8,
    delta: float = 0.1,
    ema_alpha: float = 0.1,
    update_interval_steps: int | None = None,
  ) -> dict[str, torch.Tensor]:
    del command_name, reward_name, update_interval_steps
    if env.common_step_counter > 0:
      sums = env.reward_manager.get_episode_sums(self._reward_name)[env_ids]
      weight = self._reward_cfg.weight
      if sums.numel() > 0 and weight > 0.0:
        horizon = (
          env.max_episode_length_s
          if env.reward_manager.scale_by_dt
          else float(env.max_episode_length)
        )
        batch_fraction = float((sums.mean() / (weight * horizon)).item())
        if self._ema is None:
          self._ema = batch_fraction
        else:
          self._ema = ema_alpha * batch_fraction + (1.0 - ema_alpha) * self._ema

        if env.common_step_counter - self._last_update_step >= self._interval:
          self._last_update_step = env.common_step_counter
          if self._ema > reward_fraction:
            cur_lo, cur_hi = self._command_cfg.ranges.lin_vel_x
            min_x, max_x = max_lin_vel_x
            self._command_cfg.ranges.lin_vel_x = (
              min(cur_lo, max(min_x, cur_lo - delta)),
              max(cur_hi, min(max_x, cur_hi + delta)),
            )

    lo, hi = self._command_cfg.ranges.lin_vel_x
    return {
      "lin_vel_x_min": torch.tensor(lo),
      "lin_vel_x_max": torch.tensor(hi),
      "reward_ema": torch.tensor(0.0 if self._ema is None else self._ema),
    }
