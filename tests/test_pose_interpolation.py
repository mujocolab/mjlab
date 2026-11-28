"""Regression tests for trajectory endpoint and interpolation behavior."""

import math

import torch
from conftest import get_test_device

from mjlab.utils.lab_api.math import interpolate_poses


def test_interpolate_poses_zero_steps_returns_just_endpoints():
  device = get_test_device()
  start = torch.eye(4, device=device, dtype=torch.float64)
  end = start.clone()
  end[0, 3] = 2.0

  path, num_steps = interpolate_poses(start, end, num_steps=0)

  assert num_steps == 0
  torch.testing.assert_close(path, torch.stack((start, end)))


def test_interpolate_poses_preserves_endpoints_for_small_rotations():
  device = get_test_device()
  start = torch.eye(4, device=device, dtype=torch.float64)
  end = start.clone()
  end[0, 3] = 2.0
  angle = 0.02
  end[:2, :2] = torch.tensor(
    [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
    device=device,
    dtype=end.dtype,
  )
  middle = start.clone()
  middle[0, 3] = 1.0
  middle[:2, :2] = torch.tensor(
    [
      [math.cos(angle / 2), -math.sin(angle / 2)],
      [math.sin(angle / 2), math.cos(angle / 2)],
    ],
    device=device,
    dtype=end.dtype,
  )

  path, num_steps = interpolate_poses(start, end, num_steps=1)

  assert num_steps == 1
  torch.testing.assert_close(path, torch.stack((start, middle, end)))
