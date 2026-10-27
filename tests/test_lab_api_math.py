"""Tests for mjlab.utils.lab_api.math module."""

import math

import pytest
import torch
from conftest import get_test_device

from mjlab.utils.lab_api.math import apply_delta_pose, quat_slerp, sample_gaussian


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("tau", [0.25, 0.5, 0.75])
def test_quat_slerp_preserves_inputs(device, dtype, sign, tau):
  """Shortest-path interpolation must not change either input's sign."""
  q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
  q2 = sign * torch.tensor(
    [0.5, 0.0, 0.0, math.sqrt(3) / 2], dtype=dtype, device=device
  )
  original_q1, original_q2 = q1.clone(), q2.clone()

  result = quat_slerp(q1, q2, tau)

  torch.testing.assert_close(q1, original_q1, rtol=0, atol=0)
  torch.testing.assert_close(q2, original_q2, rtol=0, atol=0)
  expected = torch.tensor(
    [math.cos(tau * math.pi / 3), 0.0, 0.0, math.sin(tau * math.pi / 3)],
    dtype=dtype,
    device=device,
  )
  torch.testing.assert_close(result, expected)
  torch.testing.assert_close(quat_slerp(q1, q2, tau), result)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quat_slerp_preserves_input_view_storage(device, dtype):
  """A row view must not silently modify the caller's quaternion buffer."""
  quaternions = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, -math.sqrt(3) / 2]],
    dtype=dtype,
    device=device,
  )
  original = quaternions.clone()

  quat_slerp(quaternions[0], quaternions[1], 0.5)

  torch.testing.assert_close(quaternions, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quat_slerp_accepts_leaf_inputs_requiring_grad(device, dtype):
  """Selecting the shortest path must not do an in-place write on a leaf."""
  q1 = torch.tensor(
    [1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device, requires_grad=True
  )
  q2 = torch.tensor(
    [-0.5, 0.0, 0.0, -math.sqrt(3) / 2],
    dtype=dtype,
    device=device,
    requires_grad=True,
  )

  result = quat_slerp(q1, q2, 0.5)
  gradients = torch.autograd.grad(result.sum(), (q1, q2))

  assert all(torch.isfinite(gradient).all() for gradient in gradients)
