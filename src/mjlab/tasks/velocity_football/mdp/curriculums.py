from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsManager
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class VisibilityBlendStage(TypedDict):
  name: str
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]
  episode_dropout_probability: float
  ball_velocity_range: tuple[float, float]
  stationary_ball_probability: float
  kick_probability: float
  kick_velocity_delta_range: tuple[float, float]
  hidden_xy_error_max: NotRequired[float]
  hidden_yaw_error_max: NotRequired[float]
  visible_xy_error_max: NotRequired[float]
  visible_yaw_error_max: NotRequired[float]
  visible_ball_control_min: NotRequired[float]
  envelope_compliance_min: NotRequired[float]
  episode_completion_min: NotRequired[float]


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

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


def scheduled_rough_terrain_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  steps_per_level: int = 24_000,
  max_level: int = 5,
  start_step: int = 0,
) -> dict[str, torch.Tensor]:
  """Sample rough-terrain levels from a fixed, reset-local training schedule."""
  if steps_per_level <= 0:
    raise ValueError("steps_per_level must be positive")
  if start_step < 0:
    raise ValueError("start_step must be non-negative")

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  available_max = terrain_origins.shape[0] - 1
  max_level = min(max_level, available_max)

  if isinstance(env_ids, slice):
    selected = torch.arange(env.num_envs, device=env.device)[env_ids]
  else:
    selected = env_ids

  elapsed_steps = max(env.common_step_counter - start_step, 0)
  current_level = min(elapsed_steps // steps_per_level + 1, max_level)
  samples = torch.rand(len(selected), device=env.device)
  if current_level == 1:
    levels = torch.where(
      samples < 0.70,
      torch.ones_like(samples, dtype=torch.long),
      torch.zeros_like(samples, dtype=torch.long),
    )
  else:
    levels = torch.full_like(samples, current_level, dtype=torch.long)
    levels = torch.where(samples >= 0.60, current_level - 1, levels)
    levels = torch.where(samples >= 0.90, 0, levels)

  terrain.terrain_levels[selected] = levels
  terrain.env_origins[selected] = terrain_origins[
    levels, terrain.terrain_types[selected]
  ]
  all_levels = terrain.terrain_levels.float()
  return {
    "current_level": torch.tensor(current_level, device=env.device),
    "mean": torch.mean(all_levels),
    "max": torch.max(all_levels),
  }


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


def lin_vel_cmd_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  reward_term_name: str = "track_linear_velocity",
  max_lin_vel_x: tuple[float, float] = (-0.5, 2.0),
  max_lin_vel_y: tuple[float, float] = (-0.5, 0.5),
  success_threshold: float = 0.7,
  range_step: float = 0.1,
) -> dict[str, torch.Tensor]:
  """Expand linear command ranges when velocity tracking reaches a threshold."""
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
  episode_reward_rate = (
    torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
    / env.max_episode_length_s
  )

  at_episode_boundary = env.common_step_counter % env.max_episode_length == 0
  if (
    at_episode_boundary and episode_reward_rate > reward_cfg.weight * success_threshold
  ):
    delta = torch.tensor((-range_step, range_step), device=env.device)
    lin_vel_x = torch.clamp(
      torch.tensor(cfg.ranges.lin_vel_x, device=env.device) + delta,
      min=max_lin_vel_x[0],
      max=max_lin_vel_x[1],
    )
    lin_vel_y = torch.clamp(
      torch.tensor(cfg.ranges.lin_vel_y, device=env.device) + delta,
      min=max_lin_vel_y[0],
      max=max_lin_vel_y[1],
    )
    cfg.ranges.lin_vel_x = (float(lin_vel_x[0]), float(lin_vel_x[1]))
    cfg.ranges.lin_vel_y = (float(lin_vel_y[0]), float(lin_vel_y[1]))

  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def push_velocity_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_term_name: str,
  max_velocity_range: dict[str, tuple[float, float]],
  unlock_command_name: str | None = None,
  unlock_lin_vel_x: tuple[float, float] | None = None,
  unlock_lin_vel_y: tuple[float, float] | None = None,
  survival_threshold: float = 0.95,
) -> dict[str, torch.Tensor]:
  """Performance-gated push curriculum for football locomotion."""
  if not 0.0 <= survival_threshold <= 1.0:
    raise ValueError("survival_threshold must be in [0, 1]")
  event_cfg = env.event_manager.get_term_cfg(event_term_name)
  start = getattr(env, "_football_push_start_range", None)
  if start is None:
    start = dict(event_cfg.params["velocity_range"])
    env._football_push_start_range = start
  progress = getattr(
    env, "_football_push_progress", torch.tensor(0.0, device=env.device)
  )
  unlocked = True
  if unlock_command_name is not None:
    if unlock_lin_vel_x is None or unlock_lin_vel_y is None:
      raise ValueError(
        "unlock_lin_vel_x and unlock_lin_vel_y are required when "
        "unlock_command_name is set"
      )
    command_term = env.command_manager.get_term(unlock_command_name)
    assert command_term is not None
    command_cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    tolerance = 1.0e-6
    unlocked = all(
      (
        command_cfg.ranges.lin_vel_x[0] <= unlock_lin_vel_x[0] + tolerance,
        command_cfg.ranges.lin_vel_x[1] >= unlock_lin_vel_x[1] - tolerance,
        command_cfg.ranges.lin_vel_y[0] <= unlock_lin_vel_y[0] + tolerance,
        command_cfg.ranges.lin_vel_y[1] >= unlock_lin_vel_y[1] - tolerance,
      )
    )
  if not unlocked:
    progress = torch.zeros_like(progress)
  selected_lengths = env.episode_length_buf[env_ids].float()
  if (
    unlocked
    and env.common_step_counter % env.max_episode_length == 0
    and torch.mean(selected_lengths) >= env.max_episode_length * survival_threshold
  ):
    progress = torch.clamp(progress + 0.1, 0.0, 1.0)
  env._football_push_progress = progress.detach()
  updated = {}
  for key, initial_range in start.items():
    initial = torch.tensor(initial_range, device=env.device)
    final = torch.tensor(max_velocity_range[key], device=env.device)
    values = torch.lerp(initial, final, progress)
    updated[key] = (float(values[0]), float(values[1]))
  event_cfg.params["velocity_range"] = updated
  return {
    "progress": progress,
    "push_x_max": torch.tensor(updated["x"][1], device=env.device),
    "unlocked": torch.tensor(float(unlocked), device=env.device),
  }


def normal_control_lin_vel_cmd_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  reward_term_name: str = "track_linear_velocity",
  ball_control_metric_name: str = "ball_control_success",
  action_acc_metric_name: str = "mean_action_acc",
  max_lin_vel_x: tuple[float, float] = (-0.5, 2.0),
  max_lin_vel_y: tuple[float, float] = (-0.5, 0.5),
  tracking_threshold: float = 0.7,
  ball_control_threshold: float = 0.3,
  survival_threshold: float = 0.7,
  action_acc_threshold: float = 0.8,
  range_step: float = 0.1,
  min_normal_episodes: int = 256,
  validation_interval_steps: int = 12_000,
  consecutive_successes: int = 3,
) -> dict[str, torch.Tensor]:
  """Expand command ranges using completed normal-control episodes only."""
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  cache_key = f"_football_normal_control_curriculum_{command_name}"
  state = cast(dict[str, Any] | None, vars(env).get(cache_key))
  if state is None:
    state = {
      "normal": torch.zeros(5, device=env.device),
      "visual": torch.zeros(3, device=env.device),
      "last_validation_step": 0,
      "successes": 0,
      "metrics": torch.zeros(6, device=env.device),
    }
    vars(env)[cache_key] = state

  if isinstance(env_ids, slice):
    selected = torch.arange(env.num_envs, device=env.device)[env_ids]
  else:
    selected = env_ids
  episode_steps = env.episode_length_buf[selected]
  completed = episode_steps > 0
  if torch.any(completed):
    selected = selected[completed]
    episode_steps = episode_steps[completed]
    visual_cache = vars(env).get("_football_masked_ball_visual")
    if isinstance(visual_cache, dict) and isinstance(
      visual_cache.get("transition_episode"), torch.Tensor
    ):
      transition_episode = visual_cache["transition_episode"][selected]
    else:
      transition_episode = torch.zeros_like(episode_steps, dtype=torch.bool)
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_scale = max(float(reward_cfg.weight), torch.finfo(torch.float32).eps)
    duration_s = episode_steps.float() * env.step_dt
    tracking = env.reward_manager._episode_sums[reward_term_name][selected]
    tracking = tracking / duration_s / reward_scale

    metrics_manager = cast(MetricsManager, env.metrics_manager)
    metric_counts = metrics_manager._step_count[selected].float().clamp(min=1.0)
    ball_control = (
      metrics_manager._episode_sums[ball_control_metric_name][selected] / metric_counts
    )
    action_acc = (
      metrics_manager._episode_sums[action_acc_metric_name][selected] / metric_counts
    )
    survival = env.reset_time_outs[selected].float()

    normal_mask = ~transition_episode
    visual_mask = transition_episode
    normal = cast(torch.Tensor, state["normal"])
    visual = cast(torch.Tensor, state["visual"])
    normal[0] += normal_mask.sum()
    normal[1] += tracking[normal_mask].sum()
    normal[2] += ball_control[normal_mask].sum()
    normal[3] += survival[normal_mask].sum()
    normal[4] += action_acc[normal_mask].sum()
    visual[0] += visual_mask.sum()
    visual[1] += tracking[visual_mask].sum()
    visual[2] += action_acc[visual_mask].sum()

  normal = cast(torch.Tensor, state["normal"])
  visual = cast(torch.Tensor, state["visual"])
  enough_steps = (
    env.common_step_counter - cast(int, state["last_validation_step"])
    >= validation_interval_steps
  )
  if enough_steps and normal[0] >= min_normal_episodes:
    normal_means = normal[1:] / normal[0]
    visual_means = visual[1:] / visual[0].clamp(min=1.0)
    metrics = cast(torch.Tensor, state["metrics"])
    metrics[:4] = normal_means
    metrics[4:] = visual_means
    passed = bool(
      (normal_means[0] > tracking_threshold)
      & (normal_means[1] > ball_control_threshold)
      & (normal_means[2] > survival_threshold)
      & (normal_means[3] < action_acc_threshold)
    )
    state["successes"] = cast(int, state["successes"]) + 1 if passed else 0
    if cast(int, state["successes"]) >= consecutive_successes:
      delta = torch.tensor((-range_step, range_step), device=env.device)
      lin_vel_x = torch.clamp(
        torch.tensor(cfg.ranges.lin_vel_x, device=env.device) + delta,
        min=max_lin_vel_x[0],
        max=max_lin_vel_x[1],
      )
      lin_vel_y = torch.clamp(
        torch.tensor(cfg.ranges.lin_vel_y, device=env.device) + delta,
        min=max_lin_vel_y[0],
        max=max_lin_vel_y[1],
      )
      cfg.ranges.lin_vel_x = (float(lin_vel_x[0]), float(lin_vel_x[1]))
      cfg.ranges.lin_vel_y = (float(lin_vel_y[0]), float(lin_vel_y[1]))
      state["successes"] = 0
    normal.zero_()
    visual.zero_()
    state["last_validation_step"] = int(env.common_step_counter)

  metrics = cast(torch.Tensor, state["metrics"])
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "normal_tracking": metrics[0],
    "normal_ball_control": metrics[1],
    "normal_survival": metrics[2],
    "normal_action_acc": metrics[3],
    "visual_tracking": metrics[4],
    "visual_action_acc": metrics[5],
    "consecutive_successes": torch.tensor(cast(int, state["successes"])),
  }


def _apply_visibility_blend_stage(
  env: ManagerBasedRlEnv,
  stage: VisibilityBlendStage,
  command_name: str,
) -> None:
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  command_cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  command_cfg.ranges.lin_vel_x = stage["lin_vel_x"]
  command_cfg.ranges.lin_vel_y = stage["lin_vel_y"]
  command_cfg.ranges.ang_vel_z = stage["ang_vel_z"]

  for term_name in ("ball_pos_b", "ball_to_feet_vectors_b", "ball_visible_mask"):
    term_cfg = env.observation_manager.get_term_cfg("actor_history", term_name)
    term_cfg.params["episode_dropout_probability"] = stage[
      "episode_dropout_probability"
    ]
  if "episode_ball_observation_hidden" in (
    env.observation_manager.active_terms.get("critic", ())
  ):
    term_cfg = env.observation_manager.get_term_cfg(
      "critic", "episode_ball_observation_hidden"
    )
    term_cfg.params["episode_dropout_probability"] = stage[
      "episode_dropout_probability"
    ]

  reset_cfg = env.event_manager.get_term_cfg("reset_football")
  reset_cfg.params["ball_velocity_range"] = stage["ball_velocity_range"]
  reset_cfg.params["stationary_ball_probability"] = stage["stationary_ball_probability"]
  kick_cfg = env.event_manager.get_term_cfg("kick_football")
  kick_cfg.params["probability"] = stage["kick_probability"]
  kick_cfg.params["velocity_delta_range"] = stage["kick_velocity_delta_range"]


def visibility_blend_task_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  stages: list[VisibilityBlendStage],
  validation_interval_steps: int = 12_000,
  consecutive_successes: int = 3,
  min_hidden_episodes: int = 32,
  min_visible_episodes: int = 128,
  hidden_xy_error_max: float = 0.25,
  hidden_yaw_error_max: float = 0.45,
  visible_xy_error_max: float = 0.30,
  visible_yaw_error_max: float = 0.50,
  visible_ball_control_min: float = 0.75,
  envelope_compliance_min: float = 0.90,
  episode_completion_min: float = 0.90,
) -> dict[str, torch.Tensor]:
  """Advance a fusion curriculum only when both observation modes succeed.

  Completed episodes are separated using the whole-episode synthetic-hidden flag.
  A visible majority therefore cannot hide a weak blind-command branch.  Each
  stage also controls command range, hidden-episode share, and football motion.
  """
  if not stages:
    raise ValueError("visibility-blend curriculum requires at least one stage")
  if validation_interval_steps <= 0 or consecutive_successes <= 0:
    raise ValueError("validation interval and success streak must be positive")

  state_key = "_visibility_blend_curriculum_state"
  state = vars(env).get(state_key)
  metric_names = (
    "user_command_error_xy",
    "user_command_error_yaw",
    "command_envelope_violation",
    "ball_control_success",
  )
  if not isinstance(state, dict):
    state = {
      "level": 0,
      "success_streak": 0,
      "next_evaluation_step": validation_interval_steps,
      "hidden_count": torch.zeros((), device=env.device),
      "visible_count": torch.zeros((), device=env.device),
      "hidden_xy_sum": torch.zeros((), device=env.device),
      "hidden_yaw_sum": torch.zeros((), device=env.device),
      "visible_xy_sum": torch.zeros((), device=env.device),
      "visible_yaw_sum": torch.zeros((), device=env.device),
      "visible_ball_sum": torch.zeros((), device=env.device),
      "envelope_sum": torch.zeros((), device=env.device),
      "completion_sum": torch.zeros((), device=env.device),
      "episode_count": torch.zeros((), device=env.device),
      "last_passed": False,
    }
    vars(env)[state_key] = state
    _apply_visibility_blend_stage(env, stages[0], command_name)

  metrics_manager = env.metrics_manager
  if all(name in metrics_manager._episode_sums for name in metric_names):
    counts = metrics_manager._step_count[env_ids].float()
    valid = counts > 0
    if torch.any(valid):
      safe_counts = torch.clamp(counts, min=1.0)
      values = {
        name: metrics_manager._episode_sums[name][env_ids] / safe_counts
        for name in metric_names
      }
      cache = vars(env).get("_football_masked_ball_visual")
      if isinstance(cache, dict) and isinstance(
        cache.get("episode_hidden"), torch.Tensor
      ):
        hidden = cache["episode_hidden"][env_ids].bool() & valid
      else:
        hidden = torch.zeros_like(valid)
      visible = ~hidden & valid

      state["hidden_count"] += hidden.sum()
      state["visible_count"] += visible.sum()
      state["hidden_xy_sum"] += values["user_command_error_xy"][hidden].sum()
      state["hidden_yaw_sum"] += values["user_command_error_yaw"][hidden].sum()
      state["visible_xy_sum"] += values["user_command_error_xy"][visible].sum()
      state["visible_yaw_sum"] += values["user_command_error_yaw"][visible].sum()
      state["visible_ball_sum"] += values["ball_control_success"][visible].sum()
      state["envelope_sum"] += values["command_envelope_violation"][valid].sum()
      state["completion_sum"] += torch.clamp(
        counts[valid] / env.max_episode_length, max=1.0
      ).sum()
      state["episode_count"] += valid.sum()

  if env.common_step_counter >= state["next_evaluation_step"]:
    current_stage = stages[state["level"]]
    hidden_count = float(state["hidden_count"].item())
    visible_count = float(state["visible_count"].item())
    episode_count = max(float(state["episode_count"].item()), 1.0)
    enough_samples = (
      hidden_count >= min_hidden_episodes and visible_count >= min_visible_episodes
    )
    hidden_denom = max(hidden_count, 1.0)
    visible_denom = max(visible_count, 1.0)
    hidden_xy = float(state["hidden_xy_sum"].item()) / hidden_denom
    hidden_yaw = float(state["hidden_yaw_sum"].item()) / hidden_denom
    visible_xy = float(state["visible_xy_sum"].item()) / visible_denom
    visible_yaw = float(state["visible_yaw_sum"].item()) / visible_denom
    ball_control = float(state["visible_ball_sum"].item()) / visible_denom
    envelope_compliance = 1.0 - float(state["envelope_sum"].item()) / episode_count
    completion = float(state["completion_sum"].item()) / episode_count
    passed = enough_samples and all(
      (
        hidden_xy <= current_stage.get("hidden_xy_error_max", hidden_xy_error_max),
        hidden_yaw <= current_stage.get("hidden_yaw_error_max", hidden_yaw_error_max),
        visible_xy <= current_stage.get("visible_xy_error_max", visible_xy_error_max),
        visible_yaw
        <= current_stage.get("visible_yaw_error_max", visible_yaw_error_max),
        ball_control
        >= current_stage.get("visible_ball_control_min", visible_ball_control_min),
        envelope_compliance
        >= current_stage.get("envelope_compliance_min", envelope_compliance_min),
        completion
        >= current_stage.get("episode_completion_min", episode_completion_min),
      )
    )
    state["last_passed"] = passed
    state["success_streak"] = state["success_streak"] + 1 if passed else 0
    if (
      state["success_streak"] >= consecutive_successes
      and state["level"] < len(stages) - 1
    ):
      state["level"] += 1
      state["success_streak"] = 0
      _apply_visibility_blend_stage(env, stages[state["level"]], command_name)

    for key in (
      "hidden_count",
      "visible_count",
      "hidden_xy_sum",
      "hidden_yaw_sum",
      "visible_xy_sum",
      "visible_yaw_sum",
      "visible_ball_sum",
      "envelope_sum",
      "completion_sum",
      "episode_count",
    ):
      state[key].zero_()
    state["next_evaluation_step"] = env.common_step_counter + validation_interval_steps

  stage = stages[state["level"]]
  return {
    "level": torch.tensor(state["level"], device=env.device),
    "success_streak": torch.tensor(state["success_streak"], device=env.device),
    "last_passed": torch.tensor(float(state["last_passed"]), device=env.device),
    "episode_dropout_probability": torch.tensor(
      stage["episode_dropout_probability"], device=env.device
    ),
    "lin_vel_x_max": torch.tensor(stage["lin_vel_x"][1], device=env.device),
    "kick_probability": torch.tensor(stage["kick_probability"], device=env.device),
    "required_ball_control": torch.tensor(
      stage.get("visible_ball_control_min", visible_ball_control_min),
      device=env.device,
    ),
    "required_envelope_compliance": torch.tensor(
      stage.get("envelope_compliance_min", envelope_compliance_min),
      device=env.device,
    ),
  }
