"""Tests for velocity-football reward functions."""

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity_football.mdp.rewards import (
  ball_front_control,
  track_ball_lin_vel_xy_exp,
)


def _make_env(*, command: torch.Tensor, robot_quat_w: torch.Tensor) -> Any:
  ball = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=torch.zeros(command.shape[0], 3))
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(command.shape[0], 3),
      root_link_quat_w=robot_quat_w,
    )
  )
  return SimpleNamespace(
    scene={"robot": robot, "ball": ball},
    command_manager=SimpleNamespace(get_command=lambda name: command),
    step_dt=0.1,
    num_envs=command.shape[0],
    device="cpu",
  )


def _make_reward(env: Any, *, period: float = 0.4, std: float = 0.5) -> Any:
  cfg = RewardTermCfg(
    func=track_ball_lin_vel_xy_exp,
    weight=1.0,
    params={"std": std, "period": period, "command_name": "twist"},
  )
  return track_ball_lin_vel_xy_exp(cfg, env)


def _identity_quat(batch_size: int = 1) -> torch.Tensor:
  quat = torch.zeros(batch_size, 4)
  quat[:, 0] = 1.0
  return quat


def test_ball_displacement_reward_averages_alternating_lateral_motion() -> None:
  env = _make_env(
    command=torch.tensor([[1.0, 0.0, 0.0]]),
    robot_quat_w=_identity_quat(),
  )
  reward = _make_reward(env)
  positions = (
    (0.0, 0.0),
    (0.1, 0.1),
    (0.2, -0.1),
    (0.3, 0.1),
    (0.4, 0.0),
  )

  values = []
  for x, y in positions:
    env.scene["ball"].data.root_link_pos_w[:] = torch.tensor([[x, y, 0.0]])
    values.append(reward(env, std=0.5, period=0.4, command_name="twist"))

  torch.testing.assert_close(values[0], torch.zeros(1))
  torch.testing.assert_close(values[-1], torch.ones(1), atol=1e-6, rtol=0.0)


def test_ball_displacement_reward_rotates_world_motion_into_robot_frame() -> None:
  sqrt_half = 2.0**-0.5
  env = _make_env(
    command=torch.tensor([[1.0, 0.0, 0.0]]),
    robot_quat_w=torch.tensor([[sqrt_half, 0.0, 0.0, sqrt_half]]),
  )
  reward = _make_reward(env)

  actual = torch.zeros(1)
  for y in (0.0, 0.1, 0.2, 0.3, 0.4):
    env.scene["ball"].data.root_link_pos_w[:] = torch.tensor([[0.0, y, 0.0]])
    actual = reward(env, std=0.5, period=0.4, command_name="twist")

  torch.testing.assert_close(actual, torch.ones(1), atol=1e-6, rtol=0.0)


def test_ball_displacement_reward_uses_available_warmup_history() -> None:
  env = _make_env(
    command=torch.tensor([[1.0, 0.0, 0.0]]),
    robot_quat_w=_identity_quat(),
  )
  reward = _make_reward(env)

  reward(env, std=0.5, period=0.4, command_name="twist")
  env.scene["ball"].data.root_link_pos_w[:, 0] = 0.05
  actual = reward(env, std=0.5, period=0.4, command_name="twist")

  torch.testing.assert_close(
    actual, torch.tensor([math.exp(-1.0)]), atol=1e-6, rtol=0.0
  )


def test_ball_displacement_reward_reset_discards_previous_episode() -> None:
  env = _make_env(
    command=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    robot_quat_w=_identity_quat(batch_size=2),
  )
  reward = _make_reward(env)

  reward(env, std=0.5, period=0.4, command_name="twist")
  env.scene["ball"].data.root_link_pos_w[:, 0] = torch.tensor([0.1, 0.1])
  reward(env, std=0.5, period=0.4, command_name="twist")
  reward.reset(torch.tensor([0]))
  env.scene["ball"].data.root_link_pos_w[:, 0] = torch.tensor([10.0, 0.2])

  actual = reward(env, std=0.5, period=0.4, command_name="twist")

  torch.testing.assert_close(actual, torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0.0)


def test_ball_front_control_uses_inclusive_hard_boundaries() -> None:
  env = _make_env(
    command=torch.zeros(7, 3),
    robot_quat_w=_identity_quat(batch_size=7),
  )
  env.scene["ball"].data.root_link_pos_w[:] = torch.tensor(
    [
      [0.25, 0.00, 0.0],
      [0.10, 0.15, 0.0],
      [0.40, -0.15, 1.0],
      [0.05, 0.00, 0.0],
      [0.45, 0.00, 0.0],
      [0.25, 0.20, 0.0],
      [-0.10, 0.00, 0.0],
    ]
  )

  actual = ball_front_control(env, x_range=(0.1, 0.4), y_abs=0.15)

  torch.testing.assert_close(actual, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))


def test_ball_front_control_follows_robot_translation_and_yaw() -> None:
  sqrt_half = 2.0**-0.5
  env = _make_env(
    command=torch.zeros(1, 3),
    robot_quat_w=torch.tensor([[sqrt_half, 0.0, 0.0, sqrt_half]]),
  )
  env.scene["robot"].data.root_link_pos_w[:] = torch.tensor([[10.0, 20.0, 0.5]])
  env.scene["ball"].data.root_link_pos_w[:] = torch.tensor([[9.9, 20.25, 0.1]])

  actual = ball_front_control(env, x_range=(0.1, 0.4), y_abs=0.15)

  torch.testing.assert_close(actual, torch.ones(1))


@pytest.mark.parametrize(
  ("x_range", "y_abs"),
  [((0.4, 0.1), 0.15), ((0.1, 0.4), 0.0), ((0.1, 0.4), -0.1)],
)
def test_ball_front_control_rejects_invalid_ranges(
  x_range: tuple[float, float], y_abs: float
) -> None:
  env = _make_env(command=torch.zeros(1, 3), robot_quat_w=_identity_quat())

  with pytest.raises(ValueError):
    ball_front_control(env, x_range=x_range, y_abs=y_abs)
