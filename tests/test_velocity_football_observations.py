"""Tests for velocity-football observation functions."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity_football.mdp.observations import (
  ball_pos_b,
  ball_to_feet_vectors_b,
  ball_vel_b,
  ball_visible_mask,
  episode_ball_observation_hidden,
  masked_ball_pos_b,
  masked_ball_to_feet_vectors_b,
  perceived_ball_pos_b,
  perceived_ball_to_feet_vectors_b,
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

  torch.testing.assert_close(ball_pos_b(env), torch.tensor([[2.0, 1.0]]))
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
    ball_pos_b(env), torch.tensor([[0.0, -1.0]]), atol=1e-6, rtol=0.0
  )
  torch.testing.assert_close(
    ball_vel_b(env), torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0.0
  )
  torch.testing.assert_close(
    ball_to_feet_vectors_b(env, asset_cfg=feet_cfg),
    torch.tensor([[-1.0, 0.0, 0.0, -1.0]]),
    atol=1e-6,
    rtol=0.0,
  )


def test_ball_derived_observations_share_the_same_perception_error() -> None:
  env = _make_env(
    robot_pos_w=torch.zeros(1, 3),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    robot_vel_w=torch.zeros(1, 3),
    feet_pos_w=torch.tensor([[[0.10, 0.05, 0.0], [0.10, -0.05, 0.0]]]),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.0]]),
    ball_vel_w=torch.zeros(1, 3),
  )
  env.num_envs = 1
  env.device = torch.device("cpu")
  env.episode_length_buf = torch.tensor([5])
  env._football_shared_ball_pos_bias = torch.tensor([[0.10, -0.02]])
  env._football_shared_ball_pos_noise = torch.tensor([[0.03, 0.04]])
  env._football_shared_ball_pos_step = torch.tensor([5])
  feet_cfg = SceneEntityCfg("robot", body_ids=[0, 1])

  position = perceived_ball_pos_b(env)
  vectors = perceived_ball_to_feet_vectors_b(env, asset_cfg=feet_cfg)

  torch.testing.assert_close(position, torch.tensor([[0.38, 0.02]]))
  torch.testing.assert_close(
    vectors,
    torch.tensor([[0.28, -0.03, 0.28, 0.07]]),
  )


def test_masked_ball_visual_terms_are_synchronized() -> None:
  env = _make_env(
    robot_pos_w=torch.zeros(2, 3),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    robot_vel_w=torch.zeros(2, 3),
    feet_pos_w=torch.tensor(
      [
        [[0.10, 0.05, 0.0], [0.10, -0.05, 0.0]],
        [[0.10, 0.05, 0.0], [0.10, -0.05, 0.0]],
      ]
    ),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.0], [1.60, 0.0, 0.0]]),
    ball_vel_w=torch.zeros(2, 3),
  )
  env.num_envs = 2
  env.device = torch.device("cpu")
  env.episode_length_buf = torch.tensor([1, 1])
  feet_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
  position = masked_ball_pos_b(
    env,
    dropout_probability=0.0,
    bias_range=0.0,
    frame_noise_range=0.0,
    asset_cfg=feet_cfg,
  )
  vectors = masked_ball_to_feet_vectors_b(
    env,
    dropout_probability=0.0,
    bias_range=0.0,
    frame_noise_range=0.0,
    asset_cfg=feet_cfg,
  )
  visible = ball_visible_mask(
    env,
    dropout_probability=0.0,
    bias_range=0.0,
    frame_noise_range=0.0,
    asset_cfg=feet_cfg,
  )

  torch.testing.assert_close(position, torch.tensor([[0.25, 0.0], [0.0, 0.0]]))
  torch.testing.assert_close(
    vectors,
    torch.tensor([[0.15, -0.05, 0.15, 0.05], [0.0, 0.0, 0.0, 0.0]]),
  )
  torch.testing.assert_close(visible, torch.tensor([[1.0], [0.0]]))


def test_episode_ball_dropout_persists_until_environment_reset() -> None:
  env = _make_env(
    robot_pos_w=torch.zeros(2, 3),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    robot_vel_w=torch.zeros(2, 3),
    feet_pos_w=torch.zeros(2, 2, 3),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
    ball_vel_w=torch.zeros(2, 3),
  )
  env.num_envs = 2
  env.device = torch.device("cpu")
  env.episode_length_buf = torch.tensor([0, 0])

  initial = ball_visible_mask(
    env,
    dropout_probability=0.0,
    episode_dropout_probability=1.0,
    bias_range=0.0,
    frame_noise_range=0.0,
  )
  torch.testing.assert_close(initial, torch.zeros(2, 1))

  env.episode_length_buf[:] = 1
  persistent = ball_visible_mask(
    env,
    dropout_probability=0.0,
    episode_dropout_probability=0.0,
    bias_range=0.0,
    frame_noise_range=0.0,
  )
  torch.testing.assert_close(persistent, torch.zeros(2, 1))

  env.episode_length_buf[:] = torch.tensor([0, 2])
  after_one_reset = ball_visible_mask(
    env,
    dropout_probability=0.0,
    episode_dropout_probability=0.0,
    bias_range=0.0,
    frame_noise_range=0.0,
  )
  torch.testing.assert_close(after_one_reset, torch.tensor([[1.0], [0.0]]))


def test_critic_hidden_flag_shares_actor_episode_dropout_sample() -> None:
  env = _make_env(
    robot_pos_w=torch.zeros(2, 3),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
    robot_vel_w=torch.zeros(2, 3),
    feet_pos_w=torch.zeros(2, 2, 3),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.0]]).repeat(2, 1),
    ball_vel_w=torch.zeros(2, 3),
  )
  env.num_envs = 2
  env.device = torch.device("cpu")
  env.episode_length_buf = torch.zeros(2, dtype=torch.long)

  critic_flag = episode_ball_observation_hidden(
    env,
    episode_dropout_probability=1.0,
    bias_range=0.0,
    frame_noise_range=0.0,
  )
  actor_mask = ball_visible_mask(
    env,
    episode_dropout_probability=1.0,
    bias_range=0.0,
    frame_noise_range=0.0,
  )

  torch.testing.assert_close(critic_flag, torch.ones(2, 1))
  torch.testing.assert_close(actor_mask, torch.zeros(2, 1))


def test_visibility_gate_falls_smoothly_after_ball_disappears() -> None:
  env = _make_env(
    robot_pos_w=torch.zeros(1, 3),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    robot_vel_w=torch.zeros(1, 3),
    feet_pos_w=torch.zeros(1, 2, 3),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.0]]),
    ball_vel_w=torch.zeros(1, 3),
  )
  env.num_envs = 1
  env.device = torch.device("cpu")
  env.episode_length_buf = torch.zeros(1, dtype=torch.long)
  ball_visible_mask(env, bias_range=0.0, frame_noise_range=0.0)
  torch.testing.assert_close(
    env._football_masked_ball_visual["visibility_gate"], torch.ones(1)
  )

  env.scene["ball"].data.root_link_pos_w[:, 0] = 2.0
  env.episode_length_buf[:] = 1
  ball_visible_mask(
    env,
    bias_range=0.0,
    frame_noise_range=0.0,
    visibility_fall_alpha=0.05,
  )
  torch.testing.assert_close(
    env._football_masked_ball_visual["visibility_gate"], torch.tensor([0.95])
  )


def test_transition_dropout_starts_mid_episode_and_recovers() -> None:
  env = _make_env(
    robot_pos_w=torch.zeros(1, 3),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    robot_vel_w=torch.zeros(1, 3),
    feet_pos_w=torch.zeros(1, 2, 3),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.0]]),
    ball_vel_w=torch.zeros(1, 3),
  )
  env.num_envs = 1
  env.device = torch.device("cpu")
  env.step_dt = 0.1
  env.episode_length_buf = torch.zeros(1, dtype=torch.long)

  def observe() -> torch.Tensor:
    return ball_visible_mask(
      env,
      bias_range=0.0,
      frame_noise_range=0.0,
      transition_dropout_probability=1.0,
      transition_dropout_start_range_s=(0.2, 0.2),
      transition_dropout_duration_range_s=(0.2, 0.2),
    )

  torch.testing.assert_close(observe(), torch.ones(1, 1))
  env.episode_length_buf[:] = 2
  torch.testing.assert_close(observe(), torch.zeros(1, 1))
  assert env._football_masked_ball_visual["synthetic_hidden"].item()
  env.episode_length_buf[:] = 4
  torch.testing.assert_close(observe(), torch.ones(1, 1))
  assert not env._football_masked_ball_visual["synthetic_hidden"].item()


def test_transition_dropout_can_remain_hidden_until_episode_end() -> None:
  env = _make_env(
    robot_pos_w=torch.zeros(1, 3),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    robot_vel_w=torch.zeros(1, 3),
    feet_pos_w=torch.zeros(1, 2, 3),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.0]]),
    ball_vel_w=torch.zeros(1, 3),
  )
  env.num_envs = 1
  env.device = torch.device("cpu")
  env.step_dt = 0.1
  env.episode_length_buf = torch.zeros(1, dtype=torch.long)

  def observe() -> torch.Tensor:
    return ball_visible_mask(
      env,
      bias_range=0.0,
      frame_noise_range=0.0,
      transition_dropout_probability=1.0,
      transition_dropout_start_range_s=(0.1, 0.1),
      transition_dropout_duration_range_s=(0.1, 0.1),
      transition_dropout_until_end_probability=1.0,
    )

  observe()
  env.episode_length_buf[:] = 100
  torch.testing.assert_close(observe(), torch.zeros(1, 1))


def test_transition_dropout_fades_sensor_reward_over_configured_time() -> None:
  env = _make_env(
    robot_pos_w=torch.zeros(1, 3),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    robot_vel_w=torch.zeros(1, 3),
    feet_pos_w=torch.zeros(1, 2, 3),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.0]]),
    ball_vel_w=torch.zeros(1, 3),
  )
  env.num_envs = 1
  env.device = torch.device("cpu")
  env.step_dt = 0.1
  env.episode_length_buf = torch.zeros(1, dtype=torch.long)

  def observe() -> torch.Tensor:
    return ball_visible_mask(
      env,
      bias_range=0.0,
      frame_noise_range=0.0,
      sensor_reward_fade_out_s=0.5,
      sensor_reward_fade_in_s=0.5,
      transition_dropout_probability=1.0,
      transition_dropout_start_range_s=(0.1, 0.1),
      transition_dropout_duration_range_s=(1.0, 1.0),
    )

  torch.testing.assert_close(observe(), torch.ones(1, 1))
  expected_gates = (0.896, 0.648, 0.352, 0.104, 0.0)
  for step, expected_gate in enumerate(expected_gates, start=1):
    env.episode_length_buf[:] = step
    torch.testing.assert_close(observe(), torch.zeros(1, 1))
    torch.testing.assert_close(
      env._football_masked_ball_visual["sensor_gate"],
      torch.tensor([expected_gate]),
      atol=1e-6,
      rtol=0.0,
    )

  torch.testing.assert_close(
    env._football_masked_ball_visual["visibility_gate"], torch.ones(1)
  )


def test_transition_dropout_excludes_standing_episodes() -> None:
  env = _make_env(
    robot_pos_w=torch.zeros(2, 3),
    robot_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
    robot_vel_w=torch.zeros(2, 3),
    feet_pos_w=torch.zeros(2, 2, 3),
    ball_pos_w=torch.tensor([[0.25, 0.0, 0.0]]).repeat(2, 1),
    ball_vel_w=torch.zeros(2, 3),
  )
  env.num_envs = 2
  env.device = torch.device("cpu")
  env.step_dt = 0.1
  env.episode_length_buf = torch.zeros(2, dtype=torch.long)
  env.command_manager = SimpleNamespace(
    get_term=lambda name: SimpleNamespace(is_standing_env=torch.tensor([True, False]))
  )

  def observe() -> torch.Tensor:
    return ball_visible_mask(
      env,
      bias_range=0.0,
      frame_noise_range=0.0,
      transition_excluded_standing_command_name="twist",
      transition_dropout_probability=1.0,
      transition_dropout_start_range_s=(0.1, 0.1),
      transition_dropout_duration_range_s=(1.0, 1.0),
      transition_dropout_until_end_probability=1.0,
    )

  torch.testing.assert_close(observe(), torch.ones(2, 1))
  env.episode_length_buf[:] = 1
  torch.testing.assert_close(observe(), torch.tensor([[1.0], [0.0]]))
  torch.testing.assert_close(
    env._football_masked_ball_visual["transition_episode"],
    torch.tensor([False, True]),
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
