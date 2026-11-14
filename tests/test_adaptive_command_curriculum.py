"""Tests for the adaptive velocity command curriculum."""

import math
from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.curriculum_manager import CurriculumManager, CurriculumTermCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel_adaptive
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg


def _make_env(
  *,
  num_envs: int = 4,
  lin_vel_x: tuple[float, float] = (-0.5, 0.5),
  reward_weight: float = 2.0,
  scale_by_dt: bool = True,
  step_dt: float = 0.02,
  episode_length_s: float = 1.0,
) -> tuple[Mock, UniformVelocityCommandCfg, torch.Tensor]:
  raw_reward = torch.zeros(num_envs)
  cmd_cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 2.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=lin_vel_x,
      lin_vel_y=(-0.5, 0.5),
      ang_vel_z=(-0.5, 0.5),
      heading=(-math.pi, math.pi),
    ),
  )
  cmd_term = Mock()
  cmd_term.cfg = cmd_cfg

  env = Mock()
  env.num_envs = num_envs
  env.device = "cpu"
  env.step_dt = step_dt
  env.max_episode_length_s = episode_length_s
  env.max_episode_length = math.ceil(episode_length_s / step_dt)
  env.common_step_counter = 0
  env.scene = {}
  env.command_manager.get_term = Mock(
    side_effect=lambda name: cmd_term if name == "twist" else None
  )
  env.reward_manager = RewardManager(
    {
      "track_linear_velocity": RewardTermCfg(
        func=lambda _env: raw_reward.clone(),
        weight=reward_weight,
      )
    },
    env,
    scale_by_dt=scale_by_dt,
  )
  return env, cmd_cfg, raw_reward


def test_initial_reset_and_empty_batches_are_noops():
  """Step 0 and empty reset batches must not seed EMA with 0 or NaN."""
  env, cmd_cfg, raw_reward = _make_env(lin_vel_x=(-0.5, 0.5))
  curriculum = CurriculumManager(
    {
      "cmd": CurriculumTermCfg(
        func=commands_vel_adaptive,
        params={
          "command_name": "twist",
          "max_lin_vel_x": (-2.0, 2.0),
          "ema_alpha": 0.5,
        },
      )
    },
    env,
  )

  # Initial reset at common_step_counter == 0.
  curriculum.compute(env_ids=torch.arange(4))
  extras = curriculum.reset()
  assert cmd_cfg.ranges.lin_vel_x == (-0.5, 0.5)
  assert extras["Curriculum/cmd/reward_ema"] == 0.0

  # Empty reset batch after stepping must not produce NaN or seed EMA.
  env.common_step_counter = 50
  curriculum.compute(env_ids=torch.empty(0, dtype=torch.long))
  extras = curriculum.reset()
  assert not math.isnan(extras["Curriculum/cmd/reward_ema"])
  assert cmd_cfg.ranges.lin_vel_x == (-0.5, 0.5)

  # First non-empty finished episode seeds EMA directly (not blended with 0).
  raw_reward[:] = 0.9
  for _ in range(env.max_episode_length):
    env.reward_manager.compute(dt=env.step_dt)
  curriculum.compute(env_ids=torch.arange(4))
  extras = curriculum.reset()
  assert extras["Curriculum/cmd/reward_ema"] == pytest.approx(0.9, rel=1e-5)
  assert cmd_cfg.ranges.lin_vel_x == pytest.approx((-0.6, 0.6))


def test_ema_accumulates_across_partial_resets_and_gates_by_interval():
  """EMA updates on intermediate resets but widens at most once per interval."""
  env, cmd_cfg, raw_reward = _make_env(
    lin_vel_x=(-0.5, 0.5), step_dt=0.1, episode_length_s=1.0
  )
  # max_episode_length == 10 steps.
  curriculum = CurriculumManager(
    {
      "cmd": CurriculumTermCfg(
        func=commands_vel_adaptive,
        params={
          "command_name": "twist",
          "max_lin_vel_x": (-1.0, 1.0),
          "reward_fraction": 0.8,
          "delta": 0.2,
          "ema_alpha": 0.5,
        },
      )
    },
    env,
  )

  # Step 5 (before interval=10): envs 0, 1 finish with full-horizon equivalent 0.9.
  raw_reward[:] = 1.8  # Over 5 of 10 steps -> 5 * 1.8 / 10 = 0.9 of full episode.
  for _ in range(5):
    env.reward_manager.compute(dt=env.step_dt)
  env.common_step_counter = 5
  curriculum.compute(env_ids=torch.tensor([0, 1]))
  env.reward_manager.reset(env_ids=torch.tensor([0, 1]))
  # Interval not yet reached -> no range expansion yet.
  assert cmd_cfg.ranges.lin_vel_x == (-0.5, 0.5)
  assert curriculum.reset()["Curriculum/cmd/reward_ema"] == pytest.approx(0.9)

  # Step 10 (interval reached): envs 2, 3 finish low -> EMA drops below 0.8.
  env.reward_manager.get_episode_sums("track_linear_velocity")[2:] = 0.0
  # Over 10 steps -> 0.4 of full episode; EMA = 0.5*0.4 + 0.5*0.9 = 0.65.
  raw_reward[:] = 0.4
  for _ in range(10):
    env.reward_manager.compute(dt=env.step_dt)
  env.common_step_counter = 10
  curriculum.compute(env_ids=torch.tensor([2, 3]))
  env.reward_manager.reset(env_ids=torch.tensor([2, 3]))
  assert curriculum.reset()["Curriculum/cmd/reward_ema"] == pytest.approx(0.65)
  assert cmd_cfg.ranges.lin_vel_x == (-0.5, 0.5)

  # Step 15 (before next interval at 20): high reward raises EMA > 0.8, still gated.
  env.reward_manager.reset()
  raw_reward[:] = 1.0
  for _ in range(10):
    env.reward_manager.compute(dt=env.step_dt)
  env.common_step_counter = 15
  curriculum.compute(env_ids=torch.arange(4))
  # EMA = 0.5 * 1.0 + 0.5 * 0.65 = 0.825 > 0.8, but 15 - 10 < 10.
  assert cmd_cfg.ranges.lin_vel_x == (-0.5, 0.5)

  # Step 20: interval elapsed -> widens once by delta=0.2.
  env.common_step_counter = 20
  curriculum.compute(env_ids=torch.arange(4))
  assert cmd_cfg.ranges.lin_vel_x == pytest.approx((-0.7, 0.7))


def test_monotonic_clamping_and_scale_by_dt_false():
  """Range caps at max_lin_vel_x, never narrows, and handles scale_by_dt=False."""
  env, cmd_cfg, raw_reward = _make_env(
    lin_vel_x=(0.0, 0.4),
    scale_by_dt=False,
    step_dt=0.1,
    episode_length_s=1.0,
  )
  curriculum = CurriculumManager(
    {
      "cmd": CurriculumTermCfg(
        func=commands_vel_adaptive,
        params={
          "command_name": "twist",
          "max_lin_vel_x": (0.0, 0.5),
          "reward_fraction": 0.8,
          "delta": 0.25,
          "ema_alpha": 1.0,
          "update_interval_steps": 10,
        },
      )
    },
    env,
  )

  # Full 10-step episode with raw_reward=0.85 and scale_by_dt=False -> fraction=0.85.
  raw_reward[:] = 0.85
  for _ in range(10):
    env.reward_manager.compute(dt=env.step_dt)
  env.common_step_counter = 10
  curriculum.compute(env_ids=torch.arange(4))
  env.reward_manager.reset()

  # Lower bound stays capped at 0.0; upper bound caps at 0.5 instead of 0.65.
  assert cmd_cfg.ranges.lin_vel_x == pytest.approx((0.0, 0.5))

  # Subsequent poor episode never narrows the range.
  raw_reward[:] = 0.1
  for _ in range(10):
    env.reward_manager.compute(dt=env.step_dt)
  env.common_step_counter = 20
  curriculum.compute(env_ids=torch.arange(4))
  assert cmd_cfg.ranges.lin_vel_x == pytest.approx((0.0, 0.5))


@pytest.mark.parametrize(
  "bad_params,exc_type",
  [
    ({"command_name": "missing", "max_lin_vel_x": (-1.0, 1.0)}, TypeError),
    (
      {
        "command_name": "twist",
        "max_lin_vel_x": (-1.0, 1.0),
        "reward_name": "missing",
      },
      ValueError,
    ),
    ({"command_name": "twist", "max_lin_vel_x": (-0.2, 1.0)}, ValueError),
    ({"command_name": "twist", "max_lin_vel_x": (-1.0, 1.0), "delta": 0.0}, ValueError),
    (
      {"command_name": "twist", "max_lin_vel_x": (-1.0, 1.0), "ema_alpha": 0.0},
      ValueError,
    ),
  ],
)
def test_commands_vel_adaptive_validates_config(bad_params, exc_type):
  """Invalid command, reward, range, delta, or EMA config fails at init."""
  env, _, _ = _make_env(lin_vel_x=(-0.5, 0.5))
  with pytest.raises(exc_type):
    CurriculumManager(
      {"cmd": CurriculumTermCfg(func=commands_vel_adaptive, params=bad_params)},
      env,
    )
