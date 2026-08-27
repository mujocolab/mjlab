"""Tests for velocity-football reset events."""

import math
from types import SimpleNamespace
from typing import Any

import torch

from mjlab.tasks.velocity_football.mdp.events import (
  kick_football_velocity,
  reset_football,
)


class _FakeEntity:
  def __init__(self, default_root_state: torch.Tensor) -> None:
    self.data = SimpleNamespace(
      default_root_state=default_root_state,
      root_link_vel_w=torch.zeros(default_root_state.shape[0], 6),
    )
    self.written_state: torch.Tensor | None = None
    self.written_env_ids: torch.Tensor | None = None
    self.written_velocity: torch.Tensor | None = None

  def write_root_state_to_sim(
    self, root_state: torch.Tensor, env_ids: torch.Tensor
  ) -> None:
    self.written_state = root_state.clone()
    self.written_env_ids = env_ids.clone()

  def write_root_link_velocity_to_sim(
    self, velocity: torch.Tensor, env_ids: torch.Tensor
  ) -> None:
    self.written_velocity = velocity.clone()
    self.written_env_ids = env_ids.clone()


class _FakeScene(dict[str, Any]):
  def __init__(self, entities: dict[str, Any], env_origins: torch.Tensor) -> None:
    super().__init__(entities)
    self.env_origins = env_origins


def _make_env(num_envs: int = 2) -> Any:
  robot_default = torch.zeros(num_envs, 13)
  robot_default[:, 2] = 0.8
  robot_default[:, 3] = 1.0
  ball_default = torch.zeros(num_envs, 13)
  ball_default[:, 3] = 1.0
  scene = _FakeScene(
    {
      "robot": _FakeEntity(robot_default),
      "ball": _FakeEntity(ball_default),
    },
    torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 2.0]])[:num_envs],
  )
  return SimpleNamespace(
    scene=scene,
    num_envs=num_envs,
    device="cpu",
  )


def test_reset_football_places_ball_relative_to_selected_robot() -> None:
  env = _make_env()
  env_ids = torch.tensor([1])

  reset_football(
    env,
    env_ids,
    ball_radius=0.1098,
    robot_xy_noise_range=(0.05, 0.05),
    robot_yaw_range=(math.pi / 2, math.pi / 2),
    ball_forward_range=(0.3, 0.3),
    ball_lateral_range=(0.1, 0.1),
    ball_velocity_range=(1.5, 1.5),
  )

  robot = env.scene["robot"]
  ball = env.scene["ball"]
  assert robot.written_state is not None
  assert ball.written_state is not None
  torch.testing.assert_close(robot.written_env_ids, env_ids)
  torch.testing.assert_close(ball.written_env_ids, env_ids)
  torch.testing.assert_close(
    robot.written_state[0, :3], torch.tensor([10.05, 20.05, 2.8])
  )
  torch.testing.assert_close(
    robot.written_state[0, 3:7],
    torch.tensor([2.0**-0.5, 0.0, 0.0, 2.0**-0.5]),
    atol=1e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(robot.written_state[0, 7:], torch.zeros(6))
  torch.testing.assert_close(
    ball.written_state[0, :3], torch.tensor([9.95, 20.35, 2.1098])
  )
  torch.testing.assert_close(
    ball.written_state[0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0])
  )
  torch.testing.assert_close(
    ball.written_state[0, 7:], torch.tensor([1.5, 1.5, 0.0, 0.0, 0.0, 0.0])
  )


def test_reset_football_uses_reference_randomization_ranges() -> None:
  num_envs = 1024
  env = _make_env(num_envs=2)
  env.scene.env_origins = torch.zeros(num_envs, 3)
  env.scene["robot"] = _FakeEntity(torch.zeros(num_envs, 13))
  env.scene["robot"].data.default_root_state[:, 2] = 0.8
  env.scene["robot"].data.default_root_state[:, 3] = 1.0
  env.scene["ball"] = _FakeEntity(torch.zeros(num_envs, 13))
  env.scene["ball"].data.default_root_state[:, 3] = 1.0
  env.num_envs = num_envs
  torch.manual_seed(7)

  reset_football(env, env_ids=None)

  robot_state = env.scene["robot"].written_state
  ball_state = env.scene["ball"].written_state
  assert robot_state is not None
  assert ball_state is not None
  assert torch.all(robot_state[:, :2].abs() <= 0.05)
  assert torch.all(ball_state[:, 7:9].abs() <= 1.5)
  torch.testing.assert_close(ball_state[:, 2], torch.full((num_envs,), 0.1098))

  relative_xy_w = ball_state[:, :2] - robot_state[:, :2]
  yaw = 2.0 * torch.atan2(robot_state[:, 6], robot_state[:, 3])
  cos_yaw = torch.cos(yaw)
  sin_yaw = torch.sin(yaw)
  forward = cos_yaw * relative_xy_w[:, 0] + sin_yaw * relative_xy_w[:, 1]
  lateral = -sin_yaw * relative_xy_w[:, 0] + cos_yaw * relative_xy_w[:, 1]
  assert torch.all((forward >= 0.1) & (forward <= 0.5))
  assert torch.all((lateral >= -0.15) & (lateral <= 0.15))


def test_reset_football_can_make_most_conservative_case_stationary() -> None:
  env = _make_env()

  reset_football(
    env,
    env_ids=None,
    ball_velocity_range=(-0.4, 0.4),
    stationary_ball_probability=1.0,
  )

  ball_state = env.scene["ball"].written_state
  assert ball_state is not None
  torch.testing.assert_close(ball_state[:, 7:9], torch.zeros(2, 2))


def test_kick_football_velocity_adds_only_planar_delta() -> None:
  env = _make_env()

  kick_football_velocity(
    env,
    env_ids=torch.tensor([0, 1]),
    probability=1.0,
    velocity_delta_range=(0.4, 0.4),
  )

  ball = env.scene["ball"]
  assert ball.written_velocity is not None
  torch.testing.assert_close(
    ball.written_velocity,
    torch.tensor(
      [[0.4, 0.4, 0.0, 0.0, 0.0, 0.0], [0.4, 0.4, 0.0, 0.0, 0.0, 0.0]]
    ),
  )
