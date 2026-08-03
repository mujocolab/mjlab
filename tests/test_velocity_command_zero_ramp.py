"""Tests for smooth moving-to-standing velocity commands."""

from types import SimpleNamespace

import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand


def _make_command() -> UniformVelocityCommand:
  command = object.__new__(UniformVelocityCommand)
  command.cfg = SimpleNamespace(
    heading_command=False,
    zero_command_ramp_time_range=(0.3, 0.5),
  )
  command._env = SimpleNamespace(step_dt=0.1)
  command.robot = SimpleNamespace()
  command.is_world_env = torch.tensor([False])
  command.is_standing_env = torch.tensor([True])
  command.zero_ramp_active = torch.tensor([True])
  command.zero_ramp_start_b = torch.tensor([[0.8, -0.4, 0.2]])
  command.zero_ramp_duration = torch.tensor([0.4])
  command.zero_ramp_elapsed = torch.tensor([0.0])
  command.vel_command_b = torch.zeros(1, 3)
  command.vel_command_w = torch.zeros(1, 3)
  return command


def test_zero_command_ramp_reaches_zero_linearly() -> None:
  command = _make_command()

  command._update_command()
  torch.testing.assert_close(
    command.vel_command_b,
    torch.tensor([[0.6, -0.3, 0.15]]),
  )

  for _ in range(3):
    command._update_command()

  torch.testing.assert_close(command.vel_command_b, torch.zeros(1, 3))
  assert not command.zero_ramp_active.item()
