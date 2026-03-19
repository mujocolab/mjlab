"""Tests for terrain normal fitting from sensor hit points."""

from unittest.mock import MagicMock, PropertyMock

import torch

from mjlab.sensor import RayCastData, RayCastSensor
from mjlab.tasks.velocity.mdp.terrain_utils import (
  fit_terrain_normal,
  terrain_normal_from_sensors,
)


def test_flat_ground():
  """Points on z=0 plane → normal = [0, 0, 1]."""
  B, N = 4, 10
  points = torch.zeros(B, N, 3)
  torch.manual_seed(0)
  points[:, :, 0] = torch.randn(B, N)
  points[:, :, 1] = torch.randn(B, N)
  valid_mask = torch.ones(B, N, dtype=torch.bool)

  normal = fit_terrain_normal(points, valid_mask)

  assert normal.shape == (B, 3)
  expected = torch.tensor([0.0, 0.0, 1.0])
  for b in range(B):
    torch.testing.assert_close(normal[b], expected, atol=1e-5, rtol=1e-5)


def test_tilted_plane():
  """Points on z = 0.5*x plane → normal perpendicular to that."""
  B, N = 2, 20
  torch.manual_seed(42)
  points = torch.zeros(B, N, 3)
  x = torch.randn(B, N)
  y = torch.randn(B, N)
  points[:, :, 0] = x
  points[:, :, 1] = y
  points[:, :, 2] = 0.5 * x
  valid_mask = torch.ones(B, N, dtype=torch.bool)

  normal = fit_terrain_normal(points, valid_mask)

  # Normal to plane z = 0.5*x is (-0.5, 0, 1) normalized.
  expected_raw = torch.tensor([-0.5, 0.0, 1.0])
  expected = expected_raw / expected_raw.norm()
  assert normal.shape == (B, 3)
  for b in range(B):
    torch.testing.assert_close(normal[b], expected, atol=1e-4, rtol=1e-4)


def test_partial_misses():
  """Half invalid points with junk z values → still correct normal."""
  B, N = 2, 20
  torch.manual_seed(42)
  points = torch.zeros(B, N, 3)
  points[:, :, 0] = torch.randn(B, N)
  points[:, :, 1] = torch.randn(B, N)
  valid_mask = torch.ones(B, N, dtype=torch.bool)
  # Invalidate even-indexed points and put junk in them.
  valid_mask[:, ::2] = False
  points[:, ::2, 2] = 999.0

  normal = fit_terrain_normal(points, valid_mask)

  expected = torch.tensor([0.0, 0.0, 1.0])
  assert normal.shape == (B, 3)
  for b in range(B):
    torch.testing.assert_close(normal[b], expected, atol=1e-4, rtol=1e-4)


def test_fewer_than_3_valid_fallback():
  """Fewer than 3 valid points (including zero) falls back to [0, 0, 1]."""
  B, N = 3, 10
  points = torch.randn(B, N, 3)
  valid_mask = torch.zeros(B, N, dtype=torch.bool)
  # Batch 0: 0 valid, batch 1: 1 valid, batch 2: 2 valid.
  valid_mask[1, 0] = True
  valid_mask[2, :2] = True

  normal = fit_terrain_normal(points, valid_mask)

  expected = torch.tensor([0.0, 0.0, 1.0])
  for b in range(B):
    torch.testing.assert_close(normal[b], expected, atol=1e-6, rtol=1e-6)


def test_collinear_points_fallback():
  """Collinear points don't define a plane, should fall back to [0, 0, 1]."""
  B, N = 2, 10
  points = torch.zeros(B, N, 3)
  # All points along the X axis.
  points[:, :, 0] = torch.linspace(0, 1, N)
  valid_mask = torch.ones(B, N, dtype=torch.bool)

  normal = fit_terrain_normal(points, valid_mask)

  expected = torch.tensor([0.0, 0.0, 1.0])
  for b in range(B):
    torch.testing.assert_close(normal[b], expected, atol=1e-6, rtol=1e-6)


def test_small_valid_plane_still_fits():
  """A tiny but planar patch should not be rejected as degenerate."""
  x_vals = torch.linspace(0.0, 1e-4, 4)
  y_vals = torch.linspace(0.0, 8e-5, 3)
  xx, yy = torch.meshgrid(x_vals, y_vals, indexing="ij")
  patch = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
  N = patch.shape[0]
  B = 2
  points = torch.zeros(B, N, 3)
  points[:, :, 0] = patch[:, 0]
  points[:, :, 1] = patch[:, 1]
  points[:, :, 2] = 0.5 * patch[:, 0]
  valid_mask = torch.ones(B, N, dtype=torch.bool)

  normal = fit_terrain_normal(points, valid_mask)

  expected_raw = torch.tensor([-0.5, 0.0, 1.0])
  expected = expected_raw / expected_raw.norm()
  for b in range(B):
    torch.testing.assert_close(normal[b], expected, atol=1e-4, rtol=1e-4)


def test_terrain_normal_from_sensors():
  """Mock env/sensors, verify subsampling + concatenation."""
  B = 4
  torch.manual_seed(7)

  # RayCastSensor mock: 100 rays, all hits on z=0.
  raycast_hit_pos = torch.zeros(B, 100, 3)
  raycast_hit_pos[:, :, 0] = torch.randn(B, 100)
  raycast_hit_pos[:, :, 1] = torch.randn(B, 100)
  raycast_distances = torch.ones(B, 100)  # all valid

  raycast_sensor = MagicMock(spec=RayCastSensor)
  raycast_data = RayCastData(
    distances=raycast_distances,
    normals_w=torch.zeros(B, 100, 3),
    hit_pos_w=raycast_hit_pos,
    pos_w=torch.zeros(B, 3),
    quat_w=torch.zeros(B, 4),
    frame_pos_w=torch.zeros(B, 1, 3),
    frame_quat_w=torch.zeros(B, 1, 4),
  )
  type(raycast_sensor).data = PropertyMock(return_value=raycast_data)

  # Mock env.
  sensors = {"raycast": raycast_sensor}
  env = MagicMock()
  env.scene.__getitem__ = MagicMock(side_effect=lambda name: sensors[name])

  normal = terrain_normal_from_sensors(
    env,
    sensor_names=("raycast",),
    max_points=16,
  )

  # 16 subsampled raycast points on z=0.
  assert normal.shape == (B, 3)
  expected = torch.tensor([0.0, 0.0, 1.0])
  for b in range(B):
    torch.testing.assert_close(normal[b], expected, atol=1e-4, rtol=1e-4)
