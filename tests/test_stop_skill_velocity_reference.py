"""Tests for the vectorized training stop-skill reference."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.tasks.velocity_football.mdp.velocity_command import (
  StopSkillVelocityReference,
  StopSkillVelocityReferenceCfg,
)


def _make_reference() -> tuple[StopSkillVelocityReference, Any]:
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene={},
    episode_length_buf=torch.zeros(1, dtype=torch.long),
    extras={"log": {}},
  )
  cfg = StopSkillVelocityReferenceCfg(
    rise_amplitude=0.2,
    rise_duration=0.3,
    fall_duration=0.3,
  )
  return StopSkillVelocityReference(cfg, cast(Any, env)), env


def _update(
  reference: StopSkillVelocityReference,
  command_x: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  return reference.update(
    torch.tensor([[command_x, 0.0, 0.0]]),
    dt=0.02,
    velocity_x_range=(-0.25, 1.0),
    velocity_y_range=(-0.25, 0.25),
  )


def test_training_stop_skill_rises_then_falls_to_zero() -> None:
  reference, _ = _make_reference()
  moving = torch.tensor([[0.7, 0.0, 0.0]])
  reference.reset(torch.tensor([0]), moving)

  first_policy, first_ball = _update(reference, command_x=0.0)
  torch.testing.assert_close(first_policy, moving[:, :2])
  torch.testing.assert_close(first_ball, moving[:, :2])

  policy_history = []
  ball_history = []
  for _ in range(40):
    policy, ball = _update(reference, command_x=0.0)
    policy_history.append(policy.clone())
    ball_history.append(ball.clone())

  policy_x = torch.cat(policy_history)[:, 0]
  ball_x = torch.cat(ball_history)[:, 0]
  assert torch.max(policy_x) > 0.89
  assert torch.all(torch.diff(ball_x) <= 1e-6)
  torch.testing.assert_close(policy_x[-1], torch.tensor(0.0), atol=1e-6, rtol=0.0)
  torch.testing.assert_close(ball_x[-1], torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_training_stop_skill_ignores_slow_command_deceleration() -> None:
  reference, _ = _make_reference()
  moving = torch.tensor([[0.8, 0.0, 0.0]])
  reference.reset(torch.tensor([0]), moving)

  for command_x in torch.linspace(0.79, 0.60, 20):
    policy, ball = _update(reference, command_x=float(command_x))
    expected = torch.tensor([[command_x, 0.0]])
    torch.testing.assert_close(policy, expected)
    torch.testing.assert_close(ball, expected)

  assert reference.state.item() == StopSkillVelocityReference.IDLE


def test_training_stop_skill_reset_does_not_depend_on_episode_length() -> None:
  reference, env = _make_reference()
  reference.state[0] = StopSkillVelocityReference.FALL
  reference.armed[0] = False
  reference.condition_count[0] = 2
  reference.elapsed[0] = 0.4
  env.episode_length_buf[0] = 37

  command = torch.tensor([[0.45, 0.0, 0.0]])
  reference.reset(torch.tensor([0]), command)

  assert reference.state.item() == StopSkillVelocityReference.IDLE
  assert reference.armed.item()
  assert reference.condition_count.item() == 0
  assert reference.elapsed.item() == 0.0
  torch.testing.assert_close(reference.reference_velocity, command[:, :2])
  torch.testing.assert_close(
    reference.command_history,
    torch.full_like(reference.command_history, 0.45),
  )
