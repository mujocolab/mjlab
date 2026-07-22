"""Tests for velocity-football observation functions."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity_football.mdp.observations import (
  ball_pos_b,
  ball_to_feet_vectors_b,
  ball_vel_b,
  phase,
)


def _make_env(
  *,
  robot_pos_w: torch.Tensor,
  robot_quat_w: torch.Tensor,
  robot_vel_w: torch.Tensor,
  feet_pos_w: torch.Tensor,
  ball_pos_w: torch.Tensor,
  ball_vel_w: torch.Tensor,
) -> Any:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=robot_pos_w,
      root_link_quat_w=robot_quat_w,
      root_link_lin_vel_w=robot_vel_w,
      body_link_pos_w=feet_pos_w,
    )
  )
  ball = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=ball_pos_w,
      root_link_lin_vel_w=ball_vel_w,
    )
  )
  return SimpleNamespace(scene={"robot": robot, "ball": ball})


def test_ball_position_and_relative_velocity_in_robot_frame() -> None:
  env = _make_env(
    robot_pos_w=torch.tensor([[10.0, 20.0, 0.5]]),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    robot_vel_w=torch.tensor([[1.0, -1.0, 0.0]]),
    feet_pos_w=torch.zeros(1, 2, 3),
    ball_pos_w=torch.tensor([[12.0, 21.0, 0.1]]),
    ball_vel_w=torch.tensor([[3.0, 1.0, 0.0]]),
  )

  torch.testing.assert_close(ball_pos_b(env), torch.tensor([[2.0, 1.0, -0.4]]))
  torch.testing.assert_close(ball_vel_b(env), torch.tensor([[2.0, 2.0, 0.0]]))


def test_ball_observations_follow_robot_yaw_and_selected_foot_order() -> None:
  sqrt_half = 2.0**-0.5
  env = _make_env(
    robot_pos_w=torch.tensor([[1.0, 2.0, 0.0]]),
    robot_quat_w=torch.tensor([[sqrt_half, 0.0, 0.0, sqrt_half]]),
    robot_vel_w=torch.zeros(1, 3),
    feet_pos_w=torch.tensor([[[2.0, 3.0, 0.0], [9.0, 9.0, 9.0], [1.0, 2.0, 0.0]]]),
    ball_pos_w=torch.tensor([[2.0, 2.0, 0.0]]),
    ball_vel_w=torch.tensor([[1.0, 0.0, 0.0]]),
  )
  feet_cfg = SceneEntityCfg("robot", body_ids=[0, 2])

  torch.testing.assert_close(
    ball_pos_b(env), torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0.0
  )
  torch.testing.assert_close(
    ball_vel_b(env), torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0.0
  )
  torch.testing.assert_close(
    ball_to_feet_vectors_b(env, asset_cfg=feet_cfg),
    torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
    atol=1e-6,
    rtol=0.0,
  )


def test_phase_follows_configured_period() -> None:
  commands = torch.ones(3, 3)
  env: Any = SimpleNamespace(
    episode_length_buf=torch.tensor([0, 15, 30]),
    step_dt=0.01,
    num_envs=3,
    device=torch.device("cpu"),
    command_manager=SimpleNamespace(get_command=lambda name: commands),
  )

  actual = phase(cast(Any, env), period=0.6, command_name="twist")

  torch.testing.assert_close(
    actual,
    torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]),
    atol=1e-6,
    rtol=0.0,
  )


def test_phase_is_zero_for_near_zero_command() -> None:
  commands = torch.tensor([[0.05, 0.05, 0.05], [0.1, 0.0, 0.0]])
  env: Any = SimpleNamespace(
    episode_length_buf=torch.tensor([15, 15]),
    step_dt=0.01,
    num_envs=2,
    device=torch.device("cpu"),
    command_manager=SimpleNamespace(get_command=lambda name: commands),
  )

  actual = phase(cast(Any, env), period=0.6, command_name="twist")

  torch.testing.assert_close(
    actual,
    torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
    atol=1e-6,
    rtol=0.0,
  )
