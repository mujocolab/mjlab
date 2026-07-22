"""Tests for velocity-football termination functions."""

from types import SimpleNamespace
from typing import Any

import torch

from mjlab.tasks.velocity_football.mdp.terminations import ball_out_of_control


class _FakeScene(dict[str, Any]):
  def __init__(self, entities: dict[str, Any], env_origins: torch.Tensor) -> None:
    super().__init__(entities)
    self.env_origins = env_origins


def _identity_quat(batch_size: int) -> torch.Tensor:
  quat = torch.zeros(batch_size, 4)
  quat[:, 0] = 1.0
  return quat


def _make_env(
  *,
  robot_pos_w: torch.Tensor,
  robot_quat_w: torch.Tensor,
  ball_pos_w: torch.Tensor,
  env_origins: torch.Tensor | None = None,
) -> Any:
  batch_size = robot_pos_w.shape[0]
  if env_origins is None:
    env_origins = torch.zeros(batch_size, 3)
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=robot_pos_w,
      root_link_quat_w=robot_quat_w,
    )
  )
  ball = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=ball_pos_w))
  scene = _FakeScene({"robot": robot, "ball": ball}, env_origins)
  return SimpleNamespace(scene=scene)


def _evaluate(env: Any) -> torch.Tensor:
  return ball_out_of_control(
    env,
    max_distance=1.5,
    min_forward=-0.5,
    max_lateral=0.5,
    max_height=0.5,
  )


def test_ball_out_of_control_checks_each_loss_condition_and_boundaries() -> None:
  ball_pos_w = torch.tensor(
    [
      [0.25, 0.00, 0.11],
      [1.51, 0.00, 0.11],
      [-0.51, 0.00, 0.11],
      [0.25, 0.51, 0.11],
      [0.25, 0.00, 0.51],
      [1.50, 0.00, 0.11],
      [-0.50, 0.00, 0.11],
      [0.00, 0.50, 0.11],
      [0.25, 0.00, 0.50],
    ]
  )
  env = _make_env(
    robot_pos_w=torch.zeros(9, 3),
    robot_quat_w=_identity_quat(9),
    ball_pos_w=ball_pos_w,
  )

  actual = _evaluate(env)

  expected = torch.tensor([False, True, True, True, True, False, False, False, False])
  torch.testing.assert_close(actual, expected)


def test_ball_out_of_control_uses_robot_frame_for_forward_limit() -> None:
  sqrt_half = 2.0**-0.5
  env = _make_env(
    robot_pos_w=torch.tensor([[10.0, 20.0, 0.0]]),
    robot_quat_w=torch.tensor([[sqrt_half, 0.0, 0.0, sqrt_half]]),
    ball_pos_w=torch.tensor([[10.0, 19.4, 0.11]]),
  )

  torch.testing.assert_close(_evaluate(env), torch.tensor([True]))


def test_ball_out_of_control_height_is_relative_to_each_environment_origin() -> None:
  env = _make_env(
    robot_pos_w=torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 10.8]]),
    robot_quat_w=_identity_quat(2),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.49], [0.25, 0.0, 10.51]]),
    env_origins=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
  )

  torch.testing.assert_close(_evaluate(env), torch.tensor([False, True]))
