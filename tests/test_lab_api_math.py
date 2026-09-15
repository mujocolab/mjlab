"""Tests for mjlab.utils.lab_api.math module."""

import pytest
import torch
from conftest import get_test_device

from mjlab.utils.lab_api.math import apply_delta_pose, sample_gaussian


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
