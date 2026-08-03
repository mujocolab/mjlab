"""Tests for the explicit ball-relative velocity reference generator."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.tasks.velocity_football.mdp.velocity_command import (
  BallRelativeVelocityReference,
  BallRelativeVelocityReferenceCfg,
)


def _make_reference() -> tuple[BallRelativeVelocityReference, Any]:
  robot_position = torch.zeros(1, 3)
  ball_position = torch.tensor([[0.25, 0.0, 0.0]])
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=robot_position,
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
  )
  ball = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=ball_position))
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene={"robot": robot, "ball": ball},
    episode_length_buf=torch.zeros(1, dtype=torch.long),
    extras={"log": {}},
  )
  cfg = BallRelativeVelocityReferenceCfg(
    fixed_position_bias_range=0.0,
    frame_position_noise_range=0.0,
  )
  return BallRelativeVelocityReference(cfg, cast(Any, env)), env


def _update(
  reference: BallRelativeVelocityReference,
  user_x: float,
) -> torch.Tensor:
  return reference.update(
    torch.tensor([[user_x, 0.0, 0.0]]),
    dt=0.1,
    velocity_x_range=(-0.25, 1.0),
    velocity_y_range=(-0.25, 0.25),
  )


def test_stop_command_preserves_a_smoothly_decaying_base_velocity() -> None:
  reference, env = _make_reference()
  moving_command = torch.tensor([[0.8, 0.0, 0.0]])
  reference.reset(torch.tensor([0]), moving_command)
  _update(reference, user_x=0.8)

  env.episode_length_buf[:] = 1
  actual = _update(reference, user_x=0.0)

  torch.testing.assert_close(actual, torch.tensor([[0.76, 0.0]]))


def test_ball_moving_ahead_can_raise_reference_before_decelerating() -> None:
  reference, env = _make_reference()
  moving_command = torch.tensor([[0.8, 0.0, 0.0]])
  reference.reset(torch.tensor([0]), moving_command)
  _update(reference, user_x=0.8)

  env.scene["ball"].data.root_link_pos_w[:, 0] = 0.80
  actual = torch.zeros(1, 2)
  for _ in range(3):
    actual = _update(reference, user_x=0.0)

  assert actual[0, 0] > moving_command[0, 0]


def test_reference_returns_to_user_command_without_residual_error() -> None:
  reference, _ = _make_reference()
  zero_command = torch.zeros(1, 3)
  reference.reset(torch.tensor([0]), zero_command)
  _update(reference, user_x=0.0)
  reference.reference_velocity[:] = torch.tensor([[0.2, 0.0]])

  actual = reference.reference_velocity
  for _ in range(10):
    actual = _update(reference, user_x=0.0)

  torch.testing.assert_close(actual, torch.zeros(1, 2), atol=1e-6, rtol=0.0)
