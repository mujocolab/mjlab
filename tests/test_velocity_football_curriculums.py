"""Tests for velocity-football command curricula."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mjlab.tasks.velocity_football.mdp.curriculums import (
  lin_vel_cmd_levels,
  normal_control_lin_vel_cmd_levels,
  scheduled_rough_terrain_levels,
)
from mjlab.tasks.velocity_football.mdp.velocity_command import (
  UniformVelocityCommandCfg,
)


def _make_env(
  reward_rate: float,
  *,
  common_step_counter: int = 1000,
  lin_vel_x: tuple[float, float] = (-0.25, 1.0),
  lin_vel_y: tuple[float, float] = (-0.25, 0.25),
) -> Any:
  command_cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(5.0, 6.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=lin_vel_x,
      lin_vel_y=lin_vel_y,
      ang_vel_z=(-1.0, 1.0),
      heading=(-3.14, 3.14),
    ),
  )
  max_episode_length_s = 20.0
  episode_sums = torch.full((4,), reward_rate * max_episode_length_s)
  reward_cfg = SimpleNamespace(weight=1.0)
  return SimpleNamespace(
    command_manager=SimpleNamespace(
      get_term=lambda name: SimpleNamespace(cfg=command_cfg)
      if name == "twist"
      else None
    ),
    reward_manager=SimpleNamespace(
      get_term_cfg=lambda name: reward_cfg,
      _episode_sums={"track_linear_velocity": episode_sums},
    ),
    common_step_counter=common_step_counter,
    max_episode_length=1000,
    max_episode_length_s=max_episode_length_s,
    device="cpu",
  )


def _run_curriculum(env: Any) -> dict[str, torch.Tensor]:
  return lin_vel_cmd_levels(
    env,
    torch.arange(4),
    command_name="twist",
    reward_term_name="track_linear_velocity",
    max_lin_vel_x=(-0.5, 2.0),
    max_lin_vel_y=(-0.5, 0.5),
    success_threshold=0.7,
    range_step=0.1,
  )


def test_scheduled_rough_terrain_curriculum_advances_every_24k_steps(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  terrain_origins = torch.zeros(6, 1, 3)
  terrain_origins[:, 0, 0] = torch.arange(6)
  terrain = SimpleNamespace(
    terrain_origins=terrain_origins,
    terrain_levels=torch.zeros(10, dtype=torch.long),
    terrain_types=torch.zeros(10, dtype=torch.long),
    env_origins=torch.zeros(10, 3),
  )
  env = SimpleNamespace(
    scene=SimpleNamespace(terrain=terrain),
    num_envs=10,
    device="cpu",
    common_step_counter=1_679_952 + 72_000,
  )
  monkeypatch.setattr(
    torch,
    "rand",
    lambda size, device: torch.tensor(
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.6, 0.7, 0.89, 0.9],
      device=device,
    ),
  )

  state = scheduled_rough_terrain_levels(
    env,
    torch.arange(10),
    steps_per_level=24_000,
    max_level=5,
    start_step=1_679_952,
  )

  assert state["current_level"].item() == 4
  assert terrain.terrain_levels.tolist() == [4, 4, 4, 4, 4, 4, 3, 3, 3, 0]
  assert torch.equal(terrain.env_origins[:, 0], terrain.terrain_levels.float())


def test_command_ranges_expand_when_tracking_reward_exceeds_threshold() -> None:
  env = _make_env(reward_rate=0.71)

  state = _run_curriculum(env)
  ranges = env.command_manager.get_term("twist").cfg.ranges

  assert ranges.lin_vel_x == pytest.approx((-0.35, 1.1))
  assert ranges.lin_vel_y == pytest.approx((-0.35, 0.35))
  assert ranges.ang_vel_z == (-1.0, 1.0)
  assert state["lin_vel_x_max"].item() == pytest.approx(1.1)


@pytest.mark.parametrize(
  ("reward_rate", "common_step_counter"),
  [(0.7, 1000), (0.71, 999)],
)
def test_command_ranges_do_not_expand_without_completed_success(
  reward_rate: float, common_step_counter: int
) -> None:
  env = _make_env(
    reward_rate=reward_rate,
    common_step_counter=common_step_counter,
  )

  _run_curriculum(env)
  ranges = env.command_manager.get_term("twist").cfg.ranges

  assert ranges.lin_vel_x == (-0.25, 1.0)
  assert ranges.lin_vel_y == (-0.25, 0.25)


def test_command_ranges_remain_clamped_at_isaac_lab_limits() -> None:
  env = _make_env(
    reward_rate=1.0,
    lin_vel_x=(-0.5, 2.0),
    lin_vel_y=(-0.5, 0.5),
  )

  _run_curriculum(env)
  ranges = env.command_manager.get_term("twist").cfg.ranges

  assert ranges.lin_vel_x == (-0.5, 2.0)
  assert ranges.lin_vel_y == (-0.5, 0.5)


def _make_normal_control_env(
  tracking: tuple[float, ...],
  *,
  transition: tuple[bool, ...],
  ball_control: tuple[float, ...] | None = None,
  survival: tuple[bool, ...] | None = None,
  action_acc: tuple[float, ...] | None = None,
) -> Any:
  num_envs = len(tracking)
  ball_control = ball_control or (1.0,) * num_envs
  survival = survival or (True,) * num_envs
  action_acc = action_acc or (0.1,) * num_envs
  command_cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(5.0, 6.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.25, 1.0),
      lin_vel_y=(-0.25, 0.25),
      ang_vel_z=(-1.0, 1.0),
      heading=(-3.14, 3.14),
    ),
  )
  command_term = SimpleNamespace(cfg=command_cfg)
  episode_steps = torch.full((num_envs,), 100, dtype=torch.long)
  step_dt = 0.02
  metric_counts = episode_steps.clone()
  return SimpleNamespace(
    command_manager=SimpleNamespace(
      get_term=lambda name: command_term if name == "twist" else None
    ),
    reward_manager=SimpleNamespace(
      get_term_cfg=lambda name: SimpleNamespace(weight=1.0),
      _episode_sums={
        "track_linear_velocity": torch.tensor(tracking) * episode_steps * step_dt
      },
    ),
    metrics_manager=SimpleNamespace(
      _step_count=metric_counts,
      _episode_sums={
        "ball_control_success": torch.tensor(ball_control) * metric_counts,
        "mean_action_acc": torch.tensor(action_acc) * metric_counts,
      },
    ),
    episode_length_buf=episode_steps,
    reset_time_outs=torch.tensor(survival),
    common_step_counter=12_001,
    num_envs=num_envs,
    step_dt=step_dt,
    device="cpu",
    _football_masked_ball_visual={
      "transition_episode": torch.tensor(transition),
    },
  )


def _run_normal_control_curriculum(env: Any) -> dict[str, torch.Tensor]:
  return normal_control_lin_vel_cmd_levels(
    env,
    torch.arange(env.num_envs),
    command_name="twist",
    min_normal_episodes=1,
    validation_interval_steps=1,
    consecutive_successes=1,
  )


def test_normal_control_curriculum_ignores_visual_episodes() -> None:
  env = _make_normal_control_env(
    (0.71, 1.0),
    transition=(False, True),
  )

  state = _run_normal_control_curriculum(env)
  ranges = env.command_manager.get_term("twist").cfg.ranges

  assert ranges.lin_vel_x == pytest.approx((-0.35, 1.1))
  assert state["normal_tracking"].item() == pytest.approx(0.71)
  assert state["visual_tracking"].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
  ("ball_control", "survival", "action_acc"),
  [((0.29,), (True,), (0.1,)), ((1.0,), (False,), (0.1,)), ((1.0,), (True,), (0.81,))],
)
def test_normal_control_curriculum_requires_all_normal_episode_gates(
  ball_control: tuple[float, ...],
  survival: tuple[bool, ...],
  action_acc: tuple[float, ...],
) -> None:
  env = _make_normal_control_env(
    (0.9,),
    transition=(False,),
    ball_control=ball_control,
    survival=survival,
    action_acc=action_acc,
  )

  _run_normal_control_curriculum(env)
  ranges = env.command_manager.get_term("twist").cfg.ranges

  assert ranges.lin_vel_x == (-0.25, 1.0)
  assert ranges.lin_vel_y == (-0.25, 0.25)
