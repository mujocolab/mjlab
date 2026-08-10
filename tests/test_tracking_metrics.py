"""Tests for motion tracking evaluation metrics."""

import math
from unittest.mock import Mock

import pytest
import torch

from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)


@pytest.fixture
def mock_command():
  """Create a mock MotionCommand for testing."""
  command = Mock()
  command.num_envs = 4
  command.device = "cpu"
  command.cfg = Mock()
  command.cfg.body_names = (
    "pelvis",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_wrist",
    "right_wrist",
  )
  return command


def test_mpkpe_zero_when_positions_match(mock_command):
  """Test MPKPE is zero when global positions are identical."""
  num_bodies = len(mock_command.cfg.body_names)
  positions = torch.rand(mock_command.num_envs, num_bodies, 3)

  mock_command.body_pos_w = positions.clone()
  mock_command.robot_body_pos_w = positions.clone()

  mpkpe = compute_mpkpe(mock_command)

  assert mpkpe.shape == (mock_command.num_envs,)
  assert torch.allclose(mpkpe, torch.zeros(mock_command.num_envs), atol=1e-6)


def test_mpkpe_correct_error(mock_command):
  """Test MPKPE computes the correct mean global error."""
  num_bodies = len(mock_command.cfg.body_names)

  mock_command.body_pos_w = torch.zeros(mock_command.num_envs, num_bodies, 3)
  mock_command.robot_body_pos_w = torch.zeros(mock_command.num_envs, num_bodies, 3)
  mock_command.robot_body_pos_w[:, :, 0] = 1.0  # 1 unit offset in x

  mpkpe = compute_mpkpe(mock_command)

  assert torch.allclose(mpkpe, torch.ones(mock_command.num_envs), atol=1e-6)


def test_mpkpe_uses_global_reference(mock_command):
  """MPKPE must read the global reference, not the drift-cancelled one.

  Pins issue #1006: setting body_pos_relative_w to match the robot exactly
  would yield zero error if it were (incorrectly) used; the metric must
  instead follow body_pos_w.
  """
  num_bodies = len(mock_command.cfg.body_names)
  robot_pos = torch.rand(mock_command.num_envs, num_bodies, 3)
  mock_command.robot_body_pos_w = robot_pos.clone()
  mock_command.body_pos_relative_w = robot_pos.clone()  # zero error if misused
  mock_command.body_pos_w = robot_pos.clone()
  mock_command.body_pos_w[:, :, 0] += 1.0  # 1 unit of global drift

  mpkpe = compute_mpkpe(mock_command)

  assert torch.allclose(mpkpe, torch.ones(mock_command.num_envs), atol=1e-6)


def test_r_mpkpe_zero_when_relative_positions_match(mock_command):
  """R-MPKPE is zero when re-anchored positions are identical."""
  num_bodies = len(mock_command.cfg.body_names)
  positions = torch.rand(mock_command.num_envs, num_bodies, 3)

  mock_command.body_pos_relative_w = positions.clone()
  mock_command.robot_body_pos_w = positions.clone()

  r_mpkpe = compute_root_relative_mpkpe(mock_command)

  assert r_mpkpe.shape == (mock_command.num_envs,)
  assert torch.allclose(r_mpkpe, torch.zeros(mock_command.num_envs), atol=1e-6)


def test_r_mpkpe_uses_relative_reference(mock_command):
  """R-MPKPE reads the re-anchored reference, not the global one.

  Setting body_pos_w to match the robot exactly would yield zero error if
  it were (incorrectly) used; the metric must instead follow
  body_pos_relative_w.
  """
  num_bodies = len(mock_command.cfg.body_names)
  robot_pos = torch.rand(mock_command.num_envs, num_bodies, 3)
  mock_command.robot_body_pos_w = robot_pos.clone()
  mock_command.body_pos_w = robot_pos.clone()  # zero error if misused
  mock_command.body_pos_relative_w = robot_pos.clone()
  mock_command.body_pos_relative_w[:, :, 0] += 1.0  # 1 unit of local pose error

  r_mpkpe = compute_root_relative_mpkpe(mock_command)

  assert torch.allclose(r_mpkpe, torch.ones(mock_command.num_envs), atol=1e-6)


def test_joint_velocity_error_rms(mock_command):
  """Joint velocity error is the per-joint RMS of the velocity error."""
  num_joints = 3

  mock_command.joint_vel = torch.zeros(mock_command.num_envs, num_joints)
  mock_command.robot_joint_vel = torch.zeros(mock_command.num_envs, num_joints)
  mock_command.robot_joint_vel[:, 0] = 3.0
  mock_command.robot_joint_vel[:, 1] = 4.0  # Error [3, 4, 0]

  error = compute_joint_velocity_error(mock_command)

  expected = math.sqrt((3.0**2 + 4.0**2 + 0.0**2) / num_joints)
  assert torch.allclose(error, torch.ones(mock_command.num_envs) * expected, atol=1e-6)


def test_ee_position_error_only_uses_specified_bodies(mock_command):
  """Test EE position error only uses specified bodies."""
  num_bodies = len(mock_command.cfg.body_names)

  mock_command.body_pos_relative_w = torch.zeros(mock_command.num_envs, num_bodies, 3)
  mock_command.robot_body_pos_w = torch.zeros(mock_command.num_envs, num_bodies, 3)

  # Large error for pelvis (not an EE).
  mock_command.robot_body_pos_w[:, 0, :] = 100.0
  # Small error for ankles.
  mock_command.robot_body_pos_w[:, 3, 0] = 1.0  # left_ankle
  mock_command.robot_body_pos_w[:, 4, 0] = 1.0  # right_ankle

  error = compute_ee_position_error(mock_command, ("left_ankle", "right_ankle"))

  # Should only reflect ankle error, not pelvis.
  assert torch.allclose(error, torch.ones(mock_command.num_envs), atol=1e-6)


def test_ee_orientation_error_detects_rotation(mock_command):
  """Test EE orientation error detects rotations."""
  num_bodies = len(mock_command.cfg.body_names)

  identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
  mock_command.body_quat_relative_w = (
    identity_quat.view(1, 1, 4).expand(mock_command.num_envs, num_bodies, 4).clone()
  )

  # 90 degree rotation around z-axis.
  rotated_quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071])
  mock_command.robot_body_quat_w = (
    rotated_quat.view(1, 1, 4).expand(mock_command.num_envs, num_bodies, 4).clone()
  )

  error = compute_ee_orientation_error(mock_command, ("left_wrist",))

  # Error should be approximately pi/2 radians.
  expected = torch.ones(mock_command.num_envs) * (3.14159 / 2)
  assert torch.allclose(error, expected, atol=0.01)


def test_ee_metrics_raise_on_unknown_body(mock_command):
  """Unknown end-effector names raise instead of silently scoring zero."""
  num_bodies = len(mock_command.cfg.body_names)
  mock_command.body_pos_relative_w = torch.zeros(mock_command.num_envs, num_bodies, 3)
  mock_command.robot_body_pos_w = torch.zeros(mock_command.num_envs, num_bodies, 3)

  with pytest.raises(ValueError, match="not tracked"):
    compute_ee_position_error(mock_command, ("nonexistent_body",))
