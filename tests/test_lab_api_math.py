"""Tests for mjlab.utils.lab_api.math module."""

import pytest
import torch
from conftest import get_test_device

from mjlab.utils.lab_api.math import (
  apply_delta_pose,
  combine_frame_transforms,
  default_orientation,
  euler_xyz_from_quat,
  matrix_from_quat,
  project_points,
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_from_matrix,
  quat_mul,
  random_orientation,
  random_yaw_orientation,
  sample_gaussian,
  unproject_depth,
  wrap_to_pi,
)


@pytest.fixture
def device():
  return get_test_device()


def test_apply_delta_pose_zero_rotation_is_finite_and_identity(device):
  """Zero rotation delta should return finite values and preserve input pose."""
  source_pos = torch.zeros(2, 3, device=device)
  source_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
  delta_pose = torch.zeros(2, 6, device=device)

  target_pos, target_rot = apply_delta_pose(source_pos, source_rot, delta_pose)

  assert torch.isfinite(target_pos).all()
  assert torch.isfinite(target_rot).all()
  assert torch.allclose(target_pos, source_pos)
  assert torch.allclose(target_rot, source_rot)


# sample_gaussian.


@pytest.mark.parametrize(
  "size, expected_shape",
  [
    (8, (8,)),
    ((4, 3), (4, 3)),
  ],
)
@pytest.mark.parametrize("as_tensor", [False, True])
def test_sample_gaussian_honors_size(device, size, expected_shape, as_tensor):
  """Output shape is always ``size``, whether mean/std are scalars or tensors."""
  if as_tensor:
    mean = torch.tensor([0.5], device=device)
    std = torch.tensor([0.05], device=device)
  else:
    mean, std = 0.5, 0.05

  out = sample_gaussian(mean, std, size, device=device)

  assert out.shape == expected_shape


def test_sample_gaussian_applies_mean_and_std(device):
  """Samples are exactly ``mean + std * noise`` for a shared noise draw."""
  mean = torch.tensor([0.5], device=device)
  std = torch.tensor([0.05], device=device)

  torch.manual_seed(0)
  out = sample_gaussian(mean, std, (128,), device=device)
  torch.manual_seed(0)
  expected = 0.5 + 0.05 * torch.randn(128, device=device)

  torch.testing.assert_close(out, expected)


def test_sample_gaussian_broadcasts_per_element_mean(device):
  """A mean shaped per-row broadcasts across the trailing dimension."""
  mean = torch.tensor([[0.0], [10.0], [20.0], [30.0]], device=device)

  out = sample_gaussian(mean, 0.0, (4, 3), device=device)

  torch.testing.assert_close(out, mean.expand(4, 3))


def test_sample_gaussian_rejects_unbroadcastable_mean(device):
  """A mean that cannot broadcast to ``size`` raises instead of resizing the output."""
  mean = torch.zeros(4, device=device)

  with pytest.raises(RuntimeError):
    sample_gaussian(mean, 1.0, (3,), device=device)


@pytest.mark.parametrize("batched", [False, True])
def test_project_points_preserves_single_batch_axis(device, batched):
  points = torch.tensor([[2.0, 4.0, 2.0], [6.0, 3.0, 3.0]], device=device)
  intrinsics = torch.tensor(
    [[2.0, 0.0, 1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0]], device=device
  )
  expected = torch.tensor([[3.0, 8.0, 2.0], [5.0, 5.0, 3.0]], device=device)
  if batched:
    points = points.unsqueeze(0)
    expected = expected.unsqueeze(0)
  torch.testing.assert_close(project_points(points, intrinsics), expected)


def test_project_points_uses_intrinsics_for_each_batch(device):
  points = torch.tensor([[[2.0, 4.0, 2.0]], [[6.0, 3.0, 3.0]]], device=device)
  intrinsics = torch.tensor(
    [
      [[2.0, 0.0, 1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0]],
      [[4.0, 0.0, 3.0], [0.0, 5.0, 1.0], [0.0, 0.0, 1.0]],
    ],
    device=device,
  )
  expected = torch.tensor([[[3.0, 8.0, 2.0]], [[11.0, 6.0, 3.0]]], device=device)
  torch.testing.assert_close(project_points(points, intrinsics), expected)


def test_quat_and_transform_helpers_eager_roundtrip(device):
  """Core quaternion, wrap, projection, and sampling helpers run eagerly."""
  roll = torch.tensor([0.2, -0.3], device=device)
  pitch = torch.tensor([0.1, 0.4], device=device)
  yaw = torch.tensor([-0.5, 0.7], device=device)
  q = quat_from_euler_xyz(roll, pitch, yaw)
  r2, p2, y2 = euler_xyz_from_quat(q)
  torch.testing.assert_close(r2, roll, atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(p2, pitch, atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(y2, yaw, atol=1e-5, rtol=1e-5)

  rot_mat = matrix_from_quat(q)
  q_recovered = quat_from_matrix(rot_mat)
  # Quaternions q and -q represent the same rotation.
  dots = (q * q_recovered).sum(dim=-1).abs()
  torch.testing.assert_close(dots, torch.ones_like(dots), atol=1e-5, rtol=1e-5)

  vec = torch.tensor([[1.0, -2.0, 0.5], [0.3, 0.4, -1.2]], device=device)
  torch.testing.assert_close(
    quat_apply_inverse(q, quat_apply(q, vec)), vec, atol=1e-5, rtol=1e-5
  )

  t_combined, q_combined = combine_frame_transforms(vec, q, torch.zeros_like(vec), q)
  torch.testing.assert_close(t_combined, vec, atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(q_combined, quat_mul(q, q), atol=1e-5, rtol=1e-5)

  wrapped = wrap_to_pi(torch.tensor([-3 * torch.pi, torch.pi, 0.5], device=device))
  torch.testing.assert_close(
    wrapped, torch.tensor([-torch.pi, torch.pi, 0.5], device=device), atol=1e-5, rtol=0
  )

  depth = torch.ones(2, 3, 4, device=device) * 2.0
  intr = torch.tensor(
    [[10.0, 0.0, 2.0], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]], device=device
  )
  pts = unproject_depth(depth, intr, is_ortho=False)
  assert pts.shape == (2, 12, 3)

  assert default_orientation(3, device).shape == (3, 4)
  assert random_orientation(3, device).shape == (3, 4)
  assert random_yaw_orientation(3, device).shape == (3, 4)
