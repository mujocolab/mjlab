"""Tests for velocity-football command curricula."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mjlab.tasks.velocity_football.mdp.curriculums import lin_vel_cmd_levels
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
