"""Tests for velocity-football reward functions."""

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity_football.mdp.rewards import (
  ball_front_control,
  ball_outside_control_zone_l2,
  command_velocity_envelope_l2,
  stop_ball_lin_vel_xy_exp,
  track_ball_lin_vel_xy_exp,
  track_ball_relative_pos_xy_exp,
  track_ball_relative_vel_xy_exp,
  track_visibility_blended_linear_velocity,
)


def _make_env(*, command: torch.Tensor, robot_quat_w: torch.Tensor) -> Any:
  ball_pos_w = torch.zeros(command.shape[0], 3)
  ball_pos_w[:, 0] = 0.25
  robot_pos_w = torch.zeros(command.shape[0], 3)
  ball = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=ball_pos_w,
      root_link_lin_vel_w=torch.zeros_like(ball_pos_w),
    )
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=robot_pos_w,
      root_link_quat_w=robot_quat_w,
      root_link_lin_vel_w=torch.zeros_like(robot_pos_w),
      root_link_lin_vel_b=torch.zeros_like(robot_pos_w),
      root_link_ang_vel_b=torch.zeros_like(robot_pos_w),
    )
  )
  return SimpleNamespace(
    scene={"robot": robot, "ball": ball},
    command_manager=SimpleNamespace(
      get_command=lambda name: command,
      get_term=lambda name: SimpleNamespace(user_command=command),
    ),
    step_dt=0.1,
    num_envs=command.shape[0],
    device="cpu",
  )


def _make_reward(env: Any, *, std: float = 0.5) -> Any:
  cfg = RewardTermCfg(
    func=track_ball_lin_vel_xy_exp,
    weight=1.0,
    params={"std": std, "command_name": "twist"},
  )
  return track_ball_lin_vel_xy_exp(cfg, env)


def _identity_quat(batch_size: int = 1) -> torch.Tensor:
  quat = torch.zeros(batch_size, 4)
  quat[:, 0] = 1.0
  return quat


def _make_relative_velocity_reward(
  env: Any,
  *,
  period: float = 0.4,
  std: float = 0.5,
) -> Any:
  cfg = RewardTermCfg(
    func=track_ball_relative_vel_xy_exp,
    weight=1.0,
    params={"std": std, "period": period},
  )
  return track_ball_relative_vel_xy_exp(cfg, env)


def test_ball_velocity_reward_tracks_instantaneous_velocity() -> None:
  env = _make_env(
    command=torch.tensor([[1.0, 0.0, 0.0]]),
    robot_quat_w=_identity_quat(),
  )
  reward = _make_reward(env)
  env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 1.0
  actual = reward(env, std=0.5, command_name="twist")
  torch.testing.assert_close(actual, torch.ones(1), atol=1e-6, rtol=0.0)


def test_ball_velocity_reward_can_disable_position_gate() -> None:
  env = _make_env(
    command=torch.tensor([[1.0, 0.0, 0.0]]),
    robot_quat_w=_identity_quat(),
  )
  reward = _make_reward(env)
  env.scene["ball"].data.root_link_pos_w[:, 0] = 2.0
  env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 1.0

  actual = reward(
    env,
    std=0.5,
    command_name="twist",
    gate_by_position=False,
  )

  torch.testing.assert_close(actual, torch.ones(1), atol=1e-6, rtol=0.0)


def test_ball_velocity_reward_rotates_world_motion_into_robot_frame() -> None:
  sqrt_half = 2.0**-0.5
  env = _make_env(
    command=torch.tensor([[1.0, 0.0, 0.0]]),
    robot_quat_w=torch.tensor([[sqrt_half, 0.0, 0.0, sqrt_half]]),
  )
  reward = _make_reward(env)
  env.scene["ball"].data.root_link_pos_w[:] = torch.tensor([[0.0, 0.25, 0.0]])
  env.scene["ball"].data.root_link_lin_vel_w[:, 1] = 1.0
  actual = reward(env, std=0.5, command_name="twist")

  torch.testing.assert_close(actual, torch.ones(1), atol=1e-6, rtol=0.0)


def test_ball_velocity_reward_penalizes_instantaneous_error() -> None:
  env = _make_env(
    command=torch.tensor([[1.0, 0.0, 0.0]]),
    robot_quat_w=_identity_quat(),
  )
  reward = _make_reward(env)
  env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 0.5
  actual = reward(env, std=0.5, command_name="twist")

  torch.testing.assert_close(
    actual, torch.tensor([math.exp(-1.0)]), atol=1e-6, rtol=0.0
  )


def test_ball_velocity_reward_can_track_unmodified_user_command() -> None:
  generated_command = torch.tensor([[0.4, 0.0, 0.0]])
  user_command = torch.tensor([[0.0, 0.0, 0.0]])
  env = _make_env(
    command=generated_command,
    robot_quat_w=_identity_quat(),
  )
  env.command_manager.get_term = lambda name: SimpleNamespace(user_command=user_command)
  reward = _make_reward(env)

  actual = reward(
    env,
    std=0.5,
    command_name="twist",
    use_user_command=True,
  )

  torch.testing.assert_close(actual, torch.ones(1), atol=1e-6, rtol=0.0)


def test_ball_velocity_reward_can_track_monotonic_ball_command() -> None:
  generated_command = torch.tensor([[0.8, 0.0, 0.0]])
  ball_command = torch.tensor([[0.2, 0.0, 0.0]])
  env = _make_env(
    command=generated_command,
    robot_quat_w=_identity_quat(),
  )
  env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 0.2
  env.command_manager.get_term = lambda name: SimpleNamespace(ball_command=ball_command)
  reward = _make_reward(env)

  actual = reward(
    env,
    std=0.5,
    command_name="twist",
    use_ball_command=True,
  )

  torch.testing.assert_close(actual, torch.ones(1), atol=1e-6, rtol=0.0)


def test_ball_velocity_reward_is_zero_without_visual_observation() -> None:
  env = _make_env(
    command=torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    robot_quat_w=_identity_quat(batch_size=2),
  )
  env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 0.5
  env._football_masked_ball_visual = {
    "visibility_gate": torch.tensor([0.0, 1.0])
  }
  reward = _make_reward(env)

  actual = reward(
    env,
    std=0.5,
    command_name="twist",
    gate_by_position=False,
    gate_by_visibility=True,
  )

  torch.testing.assert_close(actual, torch.tensor([0.0, 1.0]))


def test_visibility_blend_uses_command_when_hidden_and_bounded_recovery_when_seen() -> None:
  env = _make_env(
    command=torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    robot_quat_w=_identity_quat(batch_size=2),
  )
  env.scene["ball"].data.root_link_pos_w[:, 0] = 0.5
  env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([0.5, 0.6])
  env._football_masked_ball_visual = {
    "visibility_gate": torch.tensor([0.0, 1.0])
  }

  actual = track_visibility_blended_linear_velocity(
    env,
    std=0.5,
    command_name="twist",
  )

  torch.testing.assert_close(actual, torch.ones(2), atol=1e-6, rtol=0.0)


def test_command_velocity_envelope_has_low_speed_floor_and_high_speed_ratio() -> None:
  env = _make_env(
    command=torch.tensor([[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    robot_quat_w=_identity_quat(batch_size=2),
  )
  env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([0.25, 1.3])

  actual = command_velocity_envelope_l2(env, command_name="twist")

  torch.testing.assert_close(actual, torch.tensor([0.05**2, 0.10**2]))


def test_stop_ball_reward_is_gated_by_low_speed_command() -> None:
  command = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
  env = _make_env(command=command, robot_quat_w=_identity_quat(batch_size=2))
  env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 0.2

  actual = stop_ball_lin_vel_xy_exp(
    env,
    std=0.2,
    command_name="twist",
    command_threshold=0.1,
  )

  torch.testing.assert_close(
    actual,
    torch.tensor([math.exp(-1.0), 0.0]),
    atol=1e-6,
    rtol=0.0,
  )


def test_relative_velocity_reward_is_one_when_ball_and_robot_move_together() -> None:
  env = _make_env(command=torch.zeros(1, 3), robot_quat_w=_identity_quat())
  reward = _make_relative_velocity_reward(env)

  env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 0.5
  env.scene["robot"].data.root_link_lin_vel_w[:, 0] = 0.5
  actual = reward(env, std=0.5, period=0.4)

  torch.testing.assert_close(actual, torch.ones(1), atol=1e-6, rtol=0.0)


def test_relative_velocity_reward_penalizes_instantaneous_velocity_error() -> None:
  env = _make_env(command=torch.zeros(1, 3), robot_quat_w=_identity_quat())
  reward = _make_relative_velocity_reward(env, period=0.1, std=0.5)

  env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 0.5
  actual = reward(env, std=0.5, period=0.1)

  torch.testing.assert_close(
    actual, torch.tensor([math.exp(-1.0)]), atol=1e-6, rtol=0.0
  )


def test_relative_position_reward_uses_speed_conditioned_clipped_anchor() -> None:
  command = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [2.0, 0.0, 0.0],
    ]
  )
  env = _make_env(command=command, robot_quat_w=_identity_quat(batch_size=3))
  env.scene["ball"].data.root_link_pos_w[:] = torch.tensor(
    [
      [0.22, 0.0, 0.0],
      [0.26, 0.0, 0.0],
      [0.26, 0.0, 0.0],
    ]
  )

  actual = track_ball_relative_pos_xy_exp(
    env,
    std_x=0.07,
    std_y=0.03,
    command_name="twist",
    anchor_x=0.22,
    anchor_x_speed_gain=0.04,
    anchor_x_range=(0.22, 0.26),
  )

  torch.testing.assert_close(actual, torch.ones(3))


def test_relative_position_reward_uses_anisotropic_exponential_error() -> None:
  env = _make_env(command=torch.zeros(1, 3), robot_quat_w=_identity_quat())
  env.scene["ball"].data.root_link_pos_w[:] = torch.tensor([[0.29, 0.03, 0.0]])

  actual = track_ball_relative_pos_xy_exp(
    env,
    std_x=0.07,
    std_y=0.03,
    command_name="twist",
    anchor_x=0.22,
    anchor_x_speed_gain=0.04,
    anchor_x_range=(0.22, 0.26),
  )

  torch.testing.assert_close(
    actual, torch.tensor([math.exp(-2.0)]), atol=1e-6, rtol=0.0
  )


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


def test_ball_outside_control_zone_is_zero_inside_and_positive_outside() -> None:
  env = _make_env(command=torch.zeros(3, 3), robot_quat_w=_identity_quat(batch_size=3))
  env.scene["ball"].data.root_link_pos_w[:] = torch.tensor(
    [[0.25, 0.0, 0.0], [0.55, 0.0, 0.0], [0.25, 0.20, 0.0]]
  )
  actual = ball_outside_control_zone_l2(
    env, x_range=(0.05, 0.45), y_abs=0.15, std_x=0.10, std_y=0.05
  )
  torch.testing.assert_close(actual, torch.tensor([0.0, 1.0, 1.0]))


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
